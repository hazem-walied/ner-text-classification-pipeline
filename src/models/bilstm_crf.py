import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import tqdm


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=100, hidden_dim=128, num_layers=1, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Make a copy of tag_to_ix to avoid modifying the original
        self.tag_to_ix = tag_to_ix.copy()
        
        # Add special tags if they don't exist
        special_tags = ["<PAD>", "<START>", "<STOP>"]
        for tag in special_tags:
            if tag not in self.tag_to_ix:
                self.tag_to_ix[tag] = len(self.tag_to_ix)
                print(f"Added missing special tag: {tag} with index {self.tag_to_ix[tag]}")
        
        self.tagset_size = len(self.tag_to_ix)
        
        # Embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim // 2,  
                           num_layers=num_layers, 
                           bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        
        # Maps the output of the LSTM into tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # Matrix of transition parameters
        # transitions[i, j] is the score of transitioning from j to i
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        
        # These statements enforce constraints on the transitions:
        # Don't transition to/from padding tag
        self.transitions.data[self.tag_to_ix["<PAD>"], :] = -10000
        self.transitions.data[:, self.tag_to_ix["<PAD>"]] = -10000
        
        # Don't transition to start tag or from stop tag
        if "<START>" in self.tag_to_ix and "<STOP>" in self.tag_to_ix:
            self.transitions.data[:, self.tag_to_ix["<STOP>"]] = -10000
            self.transitions.data[self.tag_to_ix["<START>"], :] = -10000
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_lstm_features(self, input_ids, attention_mask):
        # Get sequence lengths from attention mask
        seq_lengths = attention_mask.sum(dim=1).cpu()
        
        # Embed the tokens
        embeds = self.word_embeds(input_ids)
        embeds = self.dropout(embeds)
        
        # Handle case where all sequences have zero length
        if torch.all(seq_lengths == 0):
            batch_size, max_len = input_ids.shape
            return torch.zeros(batch_size, max_len, self.tagset_size, device=input_ids.device)
        
        # Pack padded sequence for LSTM
        try:
            packed = pack_padded_sequence(embeds, seq_lengths, batch_first=True, enforce_sorted=False)
            
            # Pass through LSTM
            lstm_out, _ = self.lstm(packed)
            
            # Unpack sequence
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        except Exception as e:
            # Fallback if packing fails
            print(f"Warning: Packing failed with error: {e}. Using unpacked sequence.")
            lstm_out, _ = self.lstm(embeds)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to tag space
        lstm_feats = self.hidden2tag(lstm_out)
        
        return lstm_feats
    
    def _score_sentence(self, feats, tags, mask):
        # Gives the score of a provided tag sequence
        batch_size, seq_len, _ = feats.shape
        
        score = torch.zeros(batch_size, device=feats.device)
        
        # Add transition from start tag to first tag for each sequence
        start_tag = self.tag_to_ix.get("<START>", self.tag_to_ix["<PAD>"])
        start_tags = torch.full((batch_size, 1), start_tag, dtype=torch.long, device=feats.device)
        tags = torch.cat([start_tags, tags], dim=1)  # (batch_size, seq_len+1)
        
        for i in range(seq_len):
            # Get mask for current position (batch_size)
            mask_i = mask[:, i].bool()  # Convert to boolean tensor
            
            # Skip if all sequences are masked at this position
            if not torch.any(mask_i):
                continue
            
            # Emission score for current position
            emit_score = torch.zeros(batch_size, device=feats.device)
            valid_indices = torch.where(mask_i)[0]
            for idx in valid_indices:
                if tags[idx, i+1] < feats.size(2):  # Check if tag index is valid
                    emit_score[idx] = feats[idx, i, tags[idx, i+1]]
            
            # Transition score from previous to current tag
            trans_score = torch.zeros(batch_size, device=feats.device)
            for idx in valid_indices:
                if tags[idx, i+1] < self.transitions.size(0) and tags[idx, i] < self.transitions.size(1):
                    trans_score[idx] = self.transitions[tags[idx, i+1], tags[idx, i]]
            
            # Add both scores
            score = score + emit_score + trans_score
        
        # Add transition to STOP tag for sequences that are not fully masked
        stop_tag = self.tag_to_ix.get("<STOP>", self.tag_to_ix["<PAD>"])
        for i in range(batch_size):
            # Find the last valid position in the sequence
            last_valid = torch.sum(mask[i]).long() - 1
            if last_valid >= 0 and last_valid < seq_len:
                if tags[i, last_valid+1] < self.transitions.size(1) and stop_tag < self.transitions.size(0):
                    score[i] += self.transitions[stop_tag, tags[i, last_valid+1]]
        
        return score
    
    def _forward_alg(self, feats, mask):
        # Forward algorithm to compute partition function
        batch_size, seq_len, tagset_size = feats.shape
        
        # Initialize forward variables with -10000 (log-space)
        alphas = torch.full((batch_size, tagset_size), -10000.0, device=feats.device)
        # Start with all score from START tag or PAD tag
        start_tag = self.tag_to_ix.get("<START>", self.tag_to_ix["<PAD>"])
        alphas[:, start_tag] = 0.
        
        for i in range(seq_len):
            # Get mask for current position (batch_size)
            mask_i = mask[:, i].bool()  # Convert to boolean tensor
            
            # Skip if all sequences are masked at this position
            if not torch.any(mask_i):
                continue
            
            # (batch_size, tagset_size, 1)
            alphas_t = alphas.unsqueeze(2)
            # (batch_size, 1, tagset_size)
            emit_scores = feats[:, i].unsqueeze(1)
            
            # (batch_size, tagset_size, tagset_size)
            next_tag_var = alphas_t + self.transitions + emit_scores
            
            # Get log sum exp over the tagset_size dimension
            next_tag_var = torch.logsumexp(next_tag_var, dim=1)
            
            # Set alphas if mask is valid, otherwise keep previous value
            mask_i = mask_i.unsqueeze(1).expand_as(next_tag_var)
            alphas = torch.where(mask_i, next_tag_var, alphas)
        
        # Add transition to STOP_TAG
        stop_tag = self.tag_to_ix.get("<STOP>", self.tag_to_ix["<PAD>"])
        terminal_var = alphas + self.transitions[stop_tag]
        alphas = torch.logsumexp(terminal_var, dim=1)
        
        return alphas
    
    def neg_log_likelihood(self, input_ids, tags, attention_mask):
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(input_ids, attention_mask)
        
        # Find the best path, and the score of that path
        forward_score = self._forward_alg(feats, attention_mask)
        gold_score = self._score_sentence(feats, tags, attention_mask)
        
        # Return negative log likelihood
        return torch.mean(forward_score - gold_score)
    
    def _viterbi_decode(self, feats, mask):
        # Find the best path using Viterbi algorithm
        batch_size, seq_len, tagset_size = feats.shape
        
        # Initialize backpointers and viterbi variables
        backpointers = torch.zeros((batch_size, seq_len, tagset_size), dtype=torch.long, device=feats.device)
        
        # Initialize viterbi variables with -10000 (log-space)
        viterbi_vars = torch.full((batch_size, tagset_size), -10000.0, device=feats.device)
        start_tag = self.tag_to_ix.get("<START>", self.tag_to_ix["<PAD>"])
        viterbi_vars[:, start_tag] = 0
        
        for i in range(seq_len):
            # Get mask for current position (batch_size)
            mask_i = mask[:, i].bool()  # Convert to boolean tensor
            
            # Skip if all sequences are masked at this position
            if not torch.any(mask_i):
                continue
            
            # (batch_size, tagset_size, 1)
            viterbi_vars_t = viterbi_vars.unsqueeze(2)
            # (batch_size, tagset_size, tagset_size)
            viterbi_scores = viterbi_vars_t + self.transitions
            
            # Find the best tag for each previous tag
            # (batch_size, tagset_size)
            best_tag_id = torch.argmax(viterbi_scores, dim=1)
            best_scores = torch.gather(viterbi_scores, 1, best_tag_id.unsqueeze(1)).squeeze(1)
            
            # Add emission scores
            best_scores = best_scores + feats[:, i]
            
            # Save backpointers and best scores
            backpointers[:, i, :] = best_tag_id
            
            # Set viterbi variables if mask is valid, otherwise keep previous value
            mask_i = mask_i.unsqueeze(1).expand_as(best_scores)
            viterbi_vars = torch.where(mask_i, best_scores, viterbi_vars)
        
        # Transition to STOP_TAG
        stop_tag = self.tag_to_ix.get("<STOP>", self.tag_to_ix["<PAD>"])
        terminal_var = viterbi_vars + self.transitions[stop_tag]
        best_tag_id = torch.argmax(terminal_var, dim=1)
        
        # Follow the backpointers to decode the best path
        best_path = torch.zeros((batch_size, seq_len), dtype=torch.long, device=feats.device)
        
        # Start with the best tag for the last position
        best_path[:, -1] = best_tag_id
        
        # Follow the backpointers to find the best path
        for i in range(seq_len-2, -1, -1):
            # Get the best tag for the current position based on the next tag
            best_tag_id = torch.gather(
                backpointers[:, i+1, :], 
                1, 
                best_path[:, i+1].unsqueeze(1)
            ).squeeze(1)
            
            # Only update positions where mask is valid
            mask_i = mask[:, i+1].bool()  # Convert to boolean tensor
            best_path[:, i] = torch.where(mask_i, best_tag_id, torch.zeros_like(best_tag_id))
        
        return best_path
    
    def forward(self, input_ids, attention_mask):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(input_ids, attention_mask)
        
        # Find the best path using Viterbi algorithm
        tag_seq = self._viterbi_decode(lstm_feats, attention_mask)
        
        # For compatibility with the original code, return a list of tensors
        # where each tensor is the tag sequence for one example
        return [tag_seq[0]]
    
    def load_pretrained(self, state_dict):
        """
        Load pretrained weights even when sizes don't match perfectly.
        This handles cases where the tag dictionary or vocabulary size has changed.
        """
        model_state_dict = self.state_dict()
        pretrained_state_dict = {}
        
        # For each parameter in the model
        for name, param in model_state_dict.items():
            if name in state_dict:
                pretrained_param = state_dict[name]
                
                # Handle embedding weights
                if name == 'word_embeds.weight':
                    # Copy weights for tokens that exist in both vocabularies
                    min_vocab = min(param.shape[0], pretrained_param.shape[0])
                    param.data[:min_vocab] = pretrained_param[:min_vocab]
                    pretrained_state_dict[name] = param
                    print(f"Loaded {min_vocab}/{param.shape[0]} embedding weights")
                
                # Handle transitions matrix
                elif name == 'transitions':
                    # Copy the transitions for tags that exist in both dictionaries
                    min_tags = min(param.shape[0], pretrained_param.shape[0])
                    param.data[:min_tags, :min_tags] = pretrained_param[:min_tags, :min_tags]
                    pretrained_state_dict[name] = param
                    print(f"Loaded {min_tags}x{min_tags}/{param.shape[0]}x{param.shape[1]} transition weights")
                
                # Handle hidden2tag weights
                elif name == 'hidden2tag.weight':
                    # Copy weights for tags that exist in both dictionaries
                    min_tags = min(param.shape[0], pretrained_param.shape[0])
                    min_hidden = min(param.shape[1], pretrained_param.shape[1])
                    param.data[:min_tags, :min_hidden] = pretrained_param[:min_tags, :min_hidden]
                    pretrained_state_dict[name] = param
                    print(f"Loaded {min_tags}x{min_hidden}/{param.shape[0]}x{param.shape[1]} hidden2tag weights")
                
                # Handle hidden2tag bias
                elif name == 'hidden2tag.bias':
                    # Copy biases for tags that exist in both dictionaries
                    min_tags = min(param.shape[0], pretrained_param.shape[0])
                    param.data[:min_tags] = pretrained_param[:min_tags]
                    pretrained_state_dict[name] = param
                    print(f"Loaded {min_tags}/{param.shape[0]} hidden2tag biases")
                
                # Handle LSTM weights
                elif 'lstm' in name:
                    if param.shape == pretrained_param.shape:
                        pretrained_state_dict[name] = pretrained_param
                        print(f"Loaded {name} with shape {param.shape}")
                    else:
                        print(f"Skipped {name}: shape mismatch {param.shape} vs {pretrained_param.shape}")
                
                # For other parameters, only load if shapes match exactly
                elif param.shape == pretrained_param.shape:
                    pretrained_state_dict[name] = pretrained_param
                    print(f"Loaded {name} with shape {param.shape}")
                else:
                    print(f"Skipped {name}: shape mismatch {param.shape} vs {pretrained_param.shape}")
            else:
                print(f"Parameter {name} not found in pretrained weights")
        
        # Load the filtered state dictionary
        self.load_state_dict(pretrained_state_dict, strict=False)
        print("Loaded pretrained weights with adaptations for size mismatches")
        
        return self


def train_bilstm_crf(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            tags = batch['tags'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = model.neg_log_likelihood(input_ids, tags, attention_mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                tags = batch['tags'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                loss = model.neg_log_likelihood(input_ids, tags, attention_mask)
                
                # Update progress bar
                val_loss += loss.item()
                progress_bar.set_postfix({'loss': val_loss / (progress_bar.n + 1)})
        
        val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            print(f'New best model saved with validation loss: {val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model
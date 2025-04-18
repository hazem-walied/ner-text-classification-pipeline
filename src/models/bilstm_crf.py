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
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
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
        
        # These two statements enforce constraints on the transitions:
        # 1. Don't transition to the padding tag
        # 2. Don't transition from the padding tag
        self.transitions.data[tag_to_ix["<PAD>"], :] = -10000
        self.transitions.data[:, tag_to_ix["<PAD>"]] = -10000
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_lstm_features(self, input_ids, attention_mask):
        # Get sequence lengths from attention mask
        seq_lengths = attention_mask.sum(dim=1).cpu()
        
        # Embed the tokens
        embeds = self.word_embeds(input_ids)
        embeds = self.dropout(embeds)
        
        # Pack padded sequence for LSTM
        packed = pack_padded_sequence(embeds, seq_lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(packed)
        
        # Unpack sequence
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
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
        start_tags = torch.full((batch_size, 1), self.tag_to_ix["<PAD>"], dtype=torch.long, device=feats.device)
        tags = torch.cat([start_tags, tags], dim=1)  # (batch_size, seq_len+1)
        
        for i in range(seq_len):
            # Get mask for current position (batch_size)
            mask_i = mask[:, i]
            
            # Emission score for current position
            emit_score = torch.zeros(batch_size, device=feats.device)
            emit_score[mask_i] = feats[mask_i, i, tags[mask_i, i+1]]
            
            # Transition score from previous to current tag
            trans_score = torch.zeros(batch_size, device=feats.device)
            trans_score[mask_i] = self.transitions[tags[mask_i, i+1], tags[mask_i, i]]
            
            # Add both scores
            score = score + emit_score + trans_score
        
        return score
    
    def _forward_alg(self, feats, mask):
        # Forward algorithm to compute partition function
        batch_size, seq_len, tagset_size = feats.shape
        
        # Initialize forward variables with -10000 (log-space)
        alphas = torch.full((batch_size, tagset_size), -10000.0, device=feats.device)
        # Start with all score from <PAD>
        alphas[:, self.tag_to_ix["<PAD>"]] = 0.
        
        for i in range(seq_len):
            # Get mask for current position (batch_size)
            mask_i = mask[:, i]
            
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
        terminal_var = alphas + self.transitions[self.tag_to_ix["<PAD>"]]
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
        viterbi_vars[:, self.tag_to_ix["<PAD>"]] = 0
        
        for i in range(seq_len):
            # Get mask for current position (batch_size)
            mask_i = mask[:, i]
            
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
        terminal_var = viterbi_vars + self.transitions[self.tag_to_ix["<PAD>"]]
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
            mask_i = mask[:, i+1]
            best_path[:, i] = torch.where(mask_i, best_tag_id, torch.zeros_like(best_tag_id))
        
        return best_path
    
    def forward(self, input_ids, attention_mask):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(input_ids, attention_mask)
        
        # Find the best path using Viterbi algorithm
        tag_seq = self._viterbi_decode(lstm_feats, attention_mask)
        
        return tag_seq
    

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
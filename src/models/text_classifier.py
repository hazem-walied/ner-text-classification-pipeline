import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.5):
        super(TextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        
        # Apply mask to zero out padding
        mask = attention_mask.unsqueeze(-1).expand_as(embedded)
        embedded = embedded * mask
        
        # (batch_size, embedding_dim)
        pooled = embedded.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # (batch_size, hidden_dim)
        hidden = self.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        
        # (batch_size, num_classes)
        logits = self.fc2(hidden)
        
        return logits

def train_text_classifier(model, train_loader, val_loader, device, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'acc': correct / total
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': val_loss / (progress_bar.n + 1),
                    'acc': correct / total
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            print(f'New best model saved with validation loss: {val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model

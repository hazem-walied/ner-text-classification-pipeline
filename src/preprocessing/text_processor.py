import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import spacy
from collections import Counter
from tqdm.notebook import tqdm

# Load spaCy for tokenization and lemmatization
nlp = spacy.load("en_core_web_sm")




class TextPreprocessor:
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
    
    def preprocess(self, text):
        doc = nlp(text)
        tokens = []
        
        for token in doc:
            # Skip stopwords if configured
            if self.remove_stopwords and token.is_stop:
                continue
            
            # Skip punctuation
            if token.is_punct:
                continue
                
            # Lemmatize if configured, otherwise use the original token
            processed_token = token.lemma_ if self.lemmatize else token.text
            tokens.append(processed_token.lower())
            
        return tokens
    

class NERDataset(Dataset):
    def __init__(self, dataset_split, max_length=128):
        self.dataset = dataset_split
        self.max_length = max_length
        
        # Build vocabulary and tag dictionary
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.tag2idx = {"<PAD>": 0}
        self.idx2tag = {0: "<PAD>"}
        
        # Get tag names from dataset features
        tag_names = dataset_split.features['ner_tags'].feature.names
        for i, tag in enumerate(tag_names):
            self.tag2idx[tag] = i + 1  # +1 because 0 is for PAD
            self.idx2tag[i + 1] = tag
        
        # Build word vocabulary
        word_counter = Counter()
        for example in tqdm(dataset_split, desc="Building vocabulary"):
            for token in example['tokens']:
                word_counter[token.lower()] += 1
        
        # Keep only words that appear at least 2 times
        for word, count in word_counter.items():
            if count >= 2:
                self.word2idx[word] = len(self.word2idx)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        tokens = example['tokens']
        tags = example['ner_tags']
        
        # Convert tokens to indices
        token_indices = []
        for token in tokens[:self.max_length]:
            token = token.lower()
            if token in self.word2idx:
                token_indices.append(self.word2idx[token])
            else:
                token_indices.append(self.word2idx["<UNK>"])
        
        # Pad sequences
        padding_length = self.max_length - len(token_indices)
        if padding_length > 0:
            token_indices = token_indices + [self.word2idx["<PAD>"]] * padding_length
            tags = tags[:self.max_length] + [0] * padding_length  # 0 is PAD tag
        else:
            token_indices = token_indices[:self.max_length]
            tags = tags[:self.max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * min(len(tokens), self.max_length) + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(token_indices, dtype=torch.long),
            'tags': torch.tensor(tags, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    

class TextClassificationDataset(Dataset):
    def __init__(self, dataset_split, preprocessor=None, max_length=128):
        self.dataset = dataset_split
        self.preprocessor = preprocessor or TextPreprocessor()
        self.max_length = max_length
        
        # Build vocabulary
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        
        # Process all texts to build vocabulary
        word_counter = Counter()
        for example in tqdm(dataset_split, desc="Building vocabulary"):
            tokens = self.preprocessor.preprocess(example['text'])
            for token in tokens:
                word_counter[token] += 1
        
        # Keep only words that appear at least 5 times
        for word, count in word_counter.items():
            if count >= 5:
                self.word2idx[word] = len(self.word2idx)
        
        # Get class names
        self.num_classes = len(dataset_split.features['label'].names)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = example['text']
        label = example['label']
        
        # Preprocess text
        tokens = self.preprocessor.preprocess(text)
        
        # Convert tokens to indices
        token_indices = []
        for token in tokens[:self.max_length]:
            if token in self.word2idx:
                token_indices.append(self.word2idx[token])
            else:
                token_indices.append(self.word2idx["<UNK>"])
        
        # Pad sequences
        padding_length = self.max_length - len(token_indices)
        if padding_length > 0:
            token_indices = token_indices + [self.word2idx["<PAD>"]] * padding_length
        else:
            token_indices = token_indices[:self.max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * min(len(tokens), self.max_length) + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(token_indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def create_data_loaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader
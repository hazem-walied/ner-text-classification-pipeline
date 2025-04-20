# prepare_data.py
import torch
import pickle
import os
import sys
from datasets import load_dataset

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath('./src'))
from preprocessing.text_processor import TextPreprocessor
from models.bilstm_crf import BiLSTM_CRF
from models.text_classifier import TextClassifier

# Define paths
DATA_DIR = 'data'
VOCAB_FILE = os.path.join(DATA_DIR, 'vocab_data.pkl')

def prepare_data():
    """Extract vocabulary and other necessary data from datasets and save to disk."""
    print("Loading datasets...")
    
    # Load datasets
    ner_dataset = load_dataset("conll2003")
    text_classification_dataset = load_dataset("ag_news")
    
    print("Extracting vocabulary and processing data...")
    
    # Create preprocessor
    text_preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    # Extract NER vocabulary and tags
    ner_word2idx = {"<PAD>": 0, "<UNK>": 1}
    for example in ner_dataset['train']:
        for token in example['tokens']:
            token = token.lower()
            if token not in ner_word2idx:
                ner_word2idx[token] = len(ner_word2idx)
    
    ner_tag2idx = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, 
                  "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8}
    ner_idx2tag = {v: k for k, v in ner_tag2idx.items()}
    
    # Extract text classification vocabulary
    text_word2idx = {"<PAD>": 0, "<UNK>": 1}
    for example in text_classification_dataset['train']:
        tokens = text_preprocessor.preprocess(example['text'])
        for token in tokens:
            if token not in text_word2idx:
                text_word2idx[token] = len(text_word2idx)
    
    # Get class names
    class_names = text_classification_dataset['train'].features['label'].names
    num_classes = len(class_names)
    
    # Create directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save to disk
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump({
            'ner_word2idx': ner_word2idx,
            'ner_tag2idx': ner_tag2idx,
            'ner_idx2tag': ner_idx2tag,
            'text_word2idx': text_word2idx,
            'class_names': class_names,
            'num_classes': num_classes
        }, f)
    
    print(f"Vocabulary data saved to {VOCAB_FILE}")
    
    # Also save the preprocessor
    with open(os.path.join(DATA_DIR, 'text_preprocessor.pkl'), 'wb') as f:
        pickle.dump(text_preprocessor, f)
    
    print(f"Text preprocessor saved to {os.path.join(DATA_DIR, 'text_preprocessor.pkl')}")
    
    # Print some statistics
    print(f"NER vocabulary size: {len(ner_word2idx)}")
    print(f"Text classification vocabulary size: {len(text_word2idx)}")
    print(f"Number of classes: {num_classes}")
    
    return ner_word2idx, ner_tag2idx, text_word2idx, num_classes

if __name__ == "__main__":
    prepare_data()
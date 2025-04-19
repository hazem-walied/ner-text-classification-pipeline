import streamlit as st
import torch
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import sys
import os
sys.path.append(os.path.abspath('../src'))
from models.bilstm_crf import BiLSTM_CRF
from models.text_classifier import TextClassifier
from datasets import load_dataset
from preprocessing.text_processor import TextClassificationDataset,NERDataset,TextPreprocessor,create_data_loaders


# Load spaCy
nlp = spacy.load("en_core_web_sm")



# Load NER dataset
ner_dataset = load_dataset("conll2003")

# Load text classification dataset
text_classification_dataset = load_dataset("ag_news")

ner_train_dataset = NERDataset(ner_dataset['train'])
ner_val_dataset = NERDataset(ner_dataset['validation'])

text_preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
text_train_dataset = TextClassificationDataset(text_classification_dataset['train'], preprocessor=text_preprocessor)
text_val_dataset = TextClassificationDataset(text_classification_dataset['test'], preprocessor=text_preprocessor)




# Load models and preprocessing
@st.cache_resource
def load_models():
    # Load NER model
    ner_model = BiLSTM_CRF(
        vocab_size=len(ner_train_dataset.word2idx),
        tag_to_ix=ner_train_dataset.tag2idx,
        embedding_dim=100,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    )
    ner_model.load_state_dict(torch.load('bilstm_crf_ner.pt', map_location=torch.device('cpu')))
    ner_model.eval()
    
    # Load text classification model
    text_model = TextClassifier(
        vocab_size=len(text_train_dataset.word2idx),
        embedding_dim=100,
        hidden_dim=128,
        num_classes=text_train_dataset.num_classes,
        dropout=0.5
    )
    text_model.load_state_dict(torch.load('text_classifier.pt', map_location=torch.device('cpu')))
    text_model.eval()
    
    return ner_model, text_model

# Preprocess text for NER
def preprocess_for_ner(text, word2idx, max_length=128):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    
    # Convert tokens to indices
    token_indices = []
    for token in tokens[:max_length]:
        token = token.lower()
        if token in word2idx:
            token_indices.append(word2idx[token])
        else:
            token_indices.append(word2idx["<UNK>"])
    
    # Create attention mask
    attention_mask = [1] * len(token_indices)
    
    # Pad sequences
    padding_length = max_length - len(token_indices)
    if padding_length > 0:
        token_indices = token_indices + [word2idx["<PAD>"]] * padding_length
        attention_mask = attention_mask + [0] * padding_length
    else:
        token_indices = token_indices[:max_length]
        attention_mask = attention_mask[:max_length]
    
    return {
        'tokens': tokens[:max_length],
        'input_ids': torch.tensor(token_indices, dtype=torch.long).unsqueeze(0),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
    }

# Preprocess text for classification
def preprocess_for_classification(text, preprocessor, word2idx, max_length=128):
    # Preprocess text
    tokens = preprocessor.preprocess(text)
    
    # Convert tokens to indices
    token_indices = []
    for token in tokens[:max_length]:
        if token in word2idx:
            token_indices.append(word2idx[token])
        else:
            token_indices.append(word2idx["<UNK>"])
    
    # Create attention mask
    attention_mask = [1] * len(token_indices)
    
    # Pad sequences
    padding_length = max_length - len(token_indices)
    if padding_length > 0:
        token_indices = token_indices + [word2idx["<PAD>"]] * padding_length
        attention_mask = attention_mask + [0] * padding_length
    else:
        token_indices = token_indices[:max_length]
        attention_mask = attention_mask[:max_length]
    
    return {
        'input_ids': torch.tensor(token_indices, dtype=torch.long).unsqueeze(0),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
    }

# Visualize NER results
def visualize_ner(tokens, tags, idx2tag):
    # Create a list of tuples (token, tag)
    tagged_tokens = []
    for token, tag_idx in zip(tokens, tags[0]):
        tag = idx2tag[tag_idx.item()]
        tagged_tokens.append((token, tag))
    
    # Create HTML for visualization
    html = ""
    current_tag = "O"
    for token, tag in tagged_tokens:
        if tag != "O" and tag != current_tag:
            if current_tag != "O":
                html += "</span> "
            html += f'<span style="background-color: {get_color(tag)}; padding: 2px; border-radius: 3px;"><b>{tag}</b>: {token}'
            current_tag = tag
        elif tag != "O" and tag == current_tag:
            html += f" {token}"
        else:
            if current_tag != "O":
                html += "</span> "
            html += f"{token} "
            current_tag = "O"
    
    if current_tag != "O":
        html += "</span>"
    
    return html

# Get color for tag
def get_color(tag):
    colors = {
        "B-PER": "#FF9999",
        "I-PER": "#FF9999",
        "B-ORG": "#99FF99",
        "I-ORG": "#99FF99",
        "B-LOC": "#9999FF",
        "I-LOC": "#9999FF",
        "B-MISC": "#FFFF99",
        "I-MISC": "#FFFF99"
    }
    return colors.get(tag, "#FFFFFF")

# Plot classification probabilities
def plot_classification_probs(probs, class_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(class_names))
    
    # Horizontal bar chart
    ax.barh(y_pos, probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Probability')
    ax.set_title('Document Classification Probabilities')
    
    # Add probability values
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    
    return img

# Main app
def main():
    st.title("NER & Text Classification Pipeline")
    
    # Load models
    ner_model, text_model = load_models()
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app demonstrates a complete NLP pipeline with:"
        "\n\n"
        "- Named Entity Recognition using a custom BiLSTM-CRF model"
        "\n"
        "- Text Classification using both classical ML (TF-IDF + SVM) and neural networks"
    )
    
    # Text input
    text_input = st.text_area("Enter text for analysis:", height=200)
    
    if st.button("Analyze"):
        if text_input:
            # Process text
            with st.spinner("Processing text..."):
                # NER
                ner_processed = preprocess_for_ner(text_input, ner_train_dataset.word2idx)
                ner_tags = ner_model(ner_processed['input_ids'], ner_processed['attention_mask'])
                
                # Text Classification
                text_processed = preprocess_for_classification(
                    text_input, 
                    text_preprocessor, 
                    text_train_dataset.word2idx
                )
                with torch.no_grad():
                    logits = text_model(text_processed['input_ids'], text_processed['attention_mask'])
                    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                # Display results
                st.subheader("Named Entity Recognition")
                ner_html = visualize_ner(ner_processed['tokens'], ner_tags, ner_train_dataset.idx2tag)
                st.markdown(ner_html, unsafe_allow_html=True)
                
                st.subheader("Document Classification")
                class_names = text_classification_dataset['train'].features['label'].names
                prob_img = plot_classification_probs(probs, class_names)
                st.image(prob_img)
                
                # Show predicted class
                pred_class = class_names[np.argmax(probs)]
                st.success(f"Predicted class: {pred_class} (Confidence: {np.max(probs):.4f})")
        else:
            st.error("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
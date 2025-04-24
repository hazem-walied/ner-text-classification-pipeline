# streamlit_app/app.py
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
import pickle

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath('../src'))
from models.bilstm_crf import BiLSTM_CRF
from models.text_classifier import TextClassifier

# Define paths
DATA_DIR = '../data'
VOCAB_FILE = os.path.join(DATA_DIR, 'vocab_data.pkl')
PREPROCESSOR_FILE = os.path.join(DATA_DIR, 'text_preprocessor.pkl')
NER_MODEL_FILE = '../models/bilstm_crf_ner.pt'
TEXT_MODEL_FILE = '../models/text_classifier.pt'

# Load spaCy
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# Load vocabulary data
@st.cache_resource
def load_vocab_data():
    """Load vocabulary and other necessary data from disk."""
    if not os.path.exists(VOCAB_FILE):
        st.error(f"Vocabulary file {VOCAB_FILE} not found. Please run prepare_data.py first.")
        return None
    
    with open(VOCAB_FILE, 'rb') as f:
        data = pickle.load(f)
    
    return data

# Load text preprocessor
@st.cache_resource
def load_preprocessor():
    """Load text preprocessor from disk."""
    if not os.path.exists(PREPROCESSOR_FILE):
        st.error(f"Preprocessor file {PREPROCESSOR_FILE} not found. Please run prepare_data.py first.")
        return None
    
    with open(PREPROCESSOR_FILE, 'rb') as f:
        preprocessor = pickle.load(f)
    
    return preprocessor

# Load models
# Update the load_models function in streamlit_app/app.py

@st.cache_resource
def load_models(vocab_data):
    """Load NER and text classification models."""
    # Load NER model
    ner_model = BiLSTM_CRF(
        vocab_size=len(vocab_data['ner_word2idx']),
        tag_to_ix=vocab_data['ner_tag2idx'],
        embedding_dim=100,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    )
    
    # Load text classification model
    text_model = TextClassifier(
        vocab_size=len(vocab_data['text_word2idx']),
        embedding_dim=100,
        hidden_dim=128,
        num_classes=vocab_data['num_classes'],
        dropout=0.5
    )
    
    # Load model weights with adaptation for size mismatches
    try:
        if os.path.exists(NER_MODEL_FILE):
            state_dict = torch.load(NER_MODEL_FILE, map_location=torch.device('cpu'))
            ner_model.load_pretrained(state_dict)
            print(f"Loaded NER model from {NER_MODEL_FILE} with adaptations")
        else:
            st.warning(f"NER model file {NER_MODEL_FILE} not found. Using untrained model.")
    except Exception as e:
        st.warning(f"Could not load NER model: {e}. Using untrained model.")
    
    try:
        if os.path.exists(TEXT_MODEL_FILE):
            state_dict = torch.load(TEXT_MODEL_FILE, map_location=torch.device('cpu'))
            text_model.load_pretrained(state_dict)
            print(f"Loaded text classification model from {TEXT_MODEL_FILE} with adaptations")
        else:
            st.warning(f"Text classification model file {TEXT_MODEL_FILE} not found. Using untrained model.")
    except Exception as e:
        st.warning(f"Could not load text classification model: {e}. Using untrained model.")
    
    # Set models to evaluation mode
    ner_model.eval()
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

# Add this function to your app.py
def ensure_all_tags_exist(idx2tag, max_tag_id=15):
    """Ensure all tag IDs up to max_tag_id exist in the dictionary."""
    for i in range(max_tag_id + 1):
        if i not in idx2tag:
            # Add missing tag IDs with a default value
            if i == 9:  # Special case for <PAD>
                idx2tag[i] = "<PAD>"
            elif i == 10:  # Special case for <START>
                idx2tag[i] = "<START>"
            elif i == 11:  # Special case for <STOP>
                idx2tag[i] = "<STOP>"
            else:
                idx2tag[i] = f"Unknown-{i}"
    return idx2tag

# Update the visualize_ner function
def visualize_ner(tokens, tags, idx2tag):
    """
    Create a simple, directly aligned HTML representation of NER tags.
    Uses only the model's predictions without any predefined entity lists.
    """
    # Create HTML for visualization
    html = '<div style="line-height: 2.5; font-size: 16px;">'
    
    # First, let's identify entity spans from the model's predictions
    entity_spans = []
    current_entity = None
    start_idx = None
    
    for i, tag_idx in enumerate(tags[0]):
        tag_id = tag_idx.item() if isinstance(tag_idx, torch.Tensor) else tag_idx
        
        if tag_id in idx2tag:
            tag = idx2tag[tag_id]
        else:
            tag = "O"
        
        # Skip special tags
        if tag in ["<PAD>", "<START>", "<STOP>"] or tag.startswith("Unknown-"):
            tag = "O"
        
        # Handle B- tags (beginning of entity)
        if tag.startswith("B-"):
            if current_entity:
                entity_spans.append((start_idx, i-1, current_entity))
            current_entity = tag[2:]
            start_idx = i
        # Handle I- tags (inside entity)
        elif tag.startswith("I-"):
            if current_entity and current_entity == tag[2:]:
                # Continue current entity
                pass
            elif current_entity:
                # End previous entity and start new one
                entity_spans.append((start_idx, i-1, current_entity))
                current_entity = tag[2:]
                start_idx = i
            else:
                # Start new entity with I- tag
                current_entity = tag[2:]
                start_idx = i
        # Handle O tags (outside entity)
        elif tag == "O":
            if current_entity:
                entity_spans.append((start_idx, i-1, current_entity))
                current_entity = None
    
    # Add the last entity if there is one
    if current_entity:
        entity_spans.append((start_idx, len(tags[0])-1, current_entity))
    
    # Now render the tokens with their entity tags
    i = 0
    while i < len(tokens):
        # Check if this token is part of an entity span
        in_entity = False
        for start, end, entity_type in entity_spans:
            if start <= i <= end:
                if i == start:  # First token of the entity
                    # Get all tokens in this entity
                    entity_text = " ".join(tokens[start:end+1])
                    color = get_color(f"B-{entity_type}")
                    html += f'<span style="background-color: {color}; padding: 3px 5px; border-radius: 4px; margin: 0 2px; color: black;"><b style="color: #333;">{entity_type}</b>: {entity_text}</span> '
                    i = end + 1  # Skip to the end of the entity
                    in_entity = True
                    break
        
        if not in_entity:
            html += f'{tokens[i]} '
            i += 1
    
    html += '</div>'
    
    return html


def get_color(tag):
    """
    Return a more visually appealing color for each entity type.
    """
    colors = {
        "B-PER": "#ffcccc",  # Light red
        "I-PER": "#ffcccc",
        "B-ORG": "#ccffcc",  # Light green
        "I-ORG": "#ccffcc",
        "B-LOC": "#ccccff",  # Light blue
        "I-LOC": "#ccccff",
        "B-MISC": "#ffffcc", # Light yellow
        "I-MISC": "#ffffcc"
    }
    
    # Extract the entity type without B- or I- prefix
    if tag.startswith("B-") or tag.startswith("I-"):
        entity_type = tag[2:]
        base_tag = f"B-{entity_type}"
        return colors.get(base_tag, "#f0f0f0")
    
    return colors.get(tag, "#f0f0f0")


def consolidate_entities(tokens, tags, idx2tag):
    """
    Consolidate entity predictions to ensure consistency.
    For example, if "Microsoft" appears multiple times, it should have the same entity type.
    Returns a list of tag indices (not strings).
    """
    # Convert tag indices to tag strings
    tag_strings = []
    for tag_idx in tags[0]:
        tag_id = tag_idx.item()
        if tag_id in idx2tag:
            tag = idx2tag[tag_id]
        else:
            tag = "O"
        
        # Skip special tags
        if tag in ["<PAD>", "<START>", "<STOP>"] or tag.startswith("Unknown-"):
            tag_strings.append("O")
        else:
            tag_strings.append(tag)
    
    # Create a mapping of tokens to their most common entity type
    token_to_entity = {}
    
    # First pass: count entity types for each token
    for token, tag in zip(tokens, tag_strings):
        # Consider all tokens, not just B- tags
        if tag != "O":
            entity_type = tag[2:] if tag.startswith("B-") or tag.startswith("I-") else tag
            token_lower = token.lower()
            
            if token_lower not in token_to_entity:
                token_to_entity[token_lower] = {}
            
            if entity_type not in token_to_entity[token_lower]:
                token_to_entity[token_lower][entity_type] = 0
            
            token_to_entity[token_lower][entity_type] += 1
    
    # Determine the most common entity type for each token
    token_to_best_entity = {}
    for token, entity_counts in token_to_entity.items():
        if entity_counts:
            best_entity = max(entity_counts.items(), key=lambda x: x[1])[0]
            token_to_best_entity[token] = best_entity
    
    # Second pass: apply consistent entity types
    consolidated_tag_strings = []
    for i, (token, tag) in enumerate(zip(tokens, tag_strings)):
        token_lower = token.lower()
        
        # If this token has a consistent entity type, use it
        if token_lower in token_to_best_entity:
            # First token of its type should be B-, others I-
            is_first = True
            if i > 0:
                prev_token_lower = tokens[i-1].lower()
                if prev_token_lower in token_to_best_entity and token_to_best_entity[prev_token_lower] == token_to_best_entity[token_lower]:
                    is_first = False
            
            if is_first:
                consolidated_tag_strings.append(f"B-{token_to_best_entity[token_lower]}")
            else:
                consolidated_tag_strings.append(f"I-{token_to_best_entity[token_lower]}")
        else:
            consolidated_tag_strings.append(tag)
    
    # Create a reverse mapping from tag strings to tag indices
    tag2idx = {v: k for k, v in idx2tag.items()}
    
    # Convert consolidated tag strings back to tag indices
    consolidated_tag_indices = []
    for tag in consolidated_tag_strings:
        if tag in tag2idx:
            consolidated_tag_indices.append(tag2idx[tag])
        else:
            # Default to "O" if tag not found
            consolidated_tag_indices.append(tag2idx.get("O", 0))
    
    return consolidated_tag_indices

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
    
    # Load vocabulary data
    vocab_data = load_vocab_data()
    if vocab_data is None:
        st.error("Failed to load vocabulary data. Please run prepare_data.py first.")
        return
    
    # Ensure all tags exist in the idx2tag dictionary
    vocab_data['ner_idx2tag'] = ensure_all_tags_exist(vocab_data['ner_idx2tag'])
    
    # Load preprocessor
    preprocessor = load_preprocessor()
    if preprocessor is None:
        st.error("Failed to load text preprocessor. Please run prepare_data.py first.")
        return
    
    # Load models
    ner_model, text_model = load_models(vocab_data)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app demonstrates a complete NLP pipeline with:"
        "\n\n"
        "- Named Entity Recognition using a custom BiLSTM-CRF model"
        "\n"
        "- Text Classification using a neural network model"
    )
    
    # Text input
    text_input = st.text_area("Enter text for analysis:", height=200)
    
    if st.button("Analyze"):
        if text_input:
            # Process text
            with st.spinner("Processing text..."):
                # NER
                ner_processed = preprocess_for_ner(text_input, vocab_data['ner_word2idx'])
                ner_tags = ner_model(ner_processed['input_ids'], ner_processed['attention_mask'])
                
                # Post-process NER tags for consistency
                consolidated_tag_indices = consolidate_entities(ner_processed['tokens'], ner_tags, vocab_data['ner_idx2tag'])
                
                # Ensure tag length matches token length
                if len(consolidated_tag_indices) < len(ner_processed['tokens']):
                    # Pad with "O" tags
                    pad_length = len(ner_processed['tokens']) - len(consolidated_tag_indices)
                    consolidated_tag_indices.extend([vocab_data['ner_tag2idx'].get("O", 0)] * pad_length)
                    print(f"Padded {pad_length} tags to match token length")
                elif len(consolidated_tag_indices) > len(ner_processed['tokens']):
                    # Truncate extra tags
                    consolidated_tag_indices = consolidated_tag_indices[:len(ner_processed['tokens'])]
                    print(f"Truncated tags to match token length")
                
                # Text Classification
                text_processed = preprocess_for_classification(
                    text_input, 
                    preprocessor, 
                    vocab_data['text_word2idx']
                )
                with torch.no_grad():
                    logits = text_model(text_processed['input_ids'], text_processed['attention_mask'])
                    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                
                st.subheader("Document Classification")
                prob_img = plot_classification_probs(probs, vocab_data['class_names'])
                st.image(prob_img)
                
                # Show predicted class
                pred_class = vocab_data['class_names'][np.argmax(probs)]
                st.success(f"Predicted class: {pred_class} (Confidence: {np.max(probs):.4f})")
                
                
                # Analyze classification confidence
                confidence = np.max(probs)
                if confidence > 0.8:
                    st.write(f"High confidence classification ({confidence:.2f}): The model is very confident that this text belongs to the '{pred_class}' category.")
                elif confidence > 0.5:
                    st.write(f"Moderate confidence classification ({confidence:.2f}): The model believes this text belongs to the '{pred_class}' category, but there is some uncertainty.")
                else:
                    st.write(f"Low confidence classification ({confidence:.2f}): The model is uncertain about the classification. Consider reviewing the text or providing more context.")

                # Display results
                st.subheader("Named Entity Recognition")
                ner_html = visualize_ner(ner_processed['tokens'], [consolidated_tag_indices], vocab_data['ner_idx2tag'])
                st.markdown(ner_html, unsafe_allow_html=True)
                
                
        else:
            st.error("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
    
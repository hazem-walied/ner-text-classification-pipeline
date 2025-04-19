from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_ner_model(model, data_loader, device, idx2tag):
    model.eval()
    
    true_tags_list = []
    pred_tags_list = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            tags = batch['tags'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get predictions
            pred_tags = model(input_ids, attention_mask)
            
            # Convert to list of tags
            for i in range(len(pred_tags)):
                length = attention_mask[i].sum().item()
                
                true_tags = [idx2tag[tag.item()] for tag in tags[i][:length]]
                pred_tags_i = [idx2tag[tag.item()] for tag in pred_tags[i][:length]]
                
                true_tags_list.append(true_tags)
                pred_tags_list.append(pred_tags_i)
    
    # Calculate metrics
    report = seq_classification_report(true_tags_list, pred_tags_list)
    f1 = seq_f1_score(true_tags_list, pred_tags_list)
    
    return report, f1


def evaluate_text_classifier(model, data_loader, device, class_names):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Get predictions
            _, preds = torch.max(logits, 1)
            
            # Add to lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    report = classification_report(all_labels, all_preds, target_names=class_names)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return report, conf_matrix
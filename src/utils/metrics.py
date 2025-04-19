from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
import torch
from tqdm import tqdm

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
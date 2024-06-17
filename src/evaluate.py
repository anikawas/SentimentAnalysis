import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertForSequenceClassification

def evaluate_model():
    tokens, labels = torch.load('preprocessed_data.pt')
    
    dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('../models/bert_sentiment_model.pth'))
    model.eval()

    predictions, true_labels = [], []
    for batch in dataloader:
        with torch.no_grad():
            input_ids = batch[0]
            attention_mask = batch[1]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            true_labels.append(batch[2].cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

if __name__ == "__main__":
    evaluate_model()


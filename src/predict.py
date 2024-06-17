import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentPredictor:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, text):
        tokens = self.tokenizer.encode_plus(
            text,
            max_length=128,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
            prediction = outputs.logits.argmax(dim=1).item()
        return 'positive' if prediction == 1 else 'negative'

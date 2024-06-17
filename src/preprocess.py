#sentiment_analysis/data/smile-annotations-final.csv

import pandas as pd
from transformers import BertTokenizer
import torch
import re

# Define the preprocessing function
def preprocess(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags and entities
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Load the dataset
file_path = 'sentiment_analysis/data/smile-annotations-final.csv'
column_names = ['tweet_id', 'text', 'label']
data = pd.read_csv(file_path, names=column_names)

# Apply preprocessing to each text entry
data['cleaned_text'] = data['text'].apply(preprocess)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the cleaned text
tokens = tokenizer.batch_encode_plus(
    data['cleaned_text'].tolist(),
    max_length=128,
    padding='max_length',  # Updated padding parameter
    truncation=True,
    return_tensors='pt',
    return_attention_mask=True
)

input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# Print unique labels to debug
unique_labels = data['label'].unique()
print("Unique labels in the dataset:", unique_labels)

# Update label mapping to include all unique labels
label_mapping = {
    'nocode': 0,
    'happy': 1,
    'not-relevant': 2,
    'angry': 3,
    'disgust': 4,
    'surprise': 5,
    'sad': 6,
    'happy|surprise': 7,
    'disgust|angry': 8,
    'happy|sad': 9,
    'sad|disgust': 10,
    'sad|angry': 11,
    'sad|disgust|angry': 12
}

# Handle unexpected labels if any
for label in unique_labels:
    if label not in label_mapping:
        print(f"Unexpected label found: {label}")

# Map labels to numerical values
labels = torch.tensor([label_mapping[label] for label in data['label']])
# Ensure no unmapped labels exist
if (labels == -1).any():
    raise ValueError("There are labels that were not mapped correctly.")

# Verify sizes
print("Input IDs size:", input_ids.size())
print("Attention Mask size:", attention_mask.size())
print("Labels size:", labels.size())

# Save preprocessed data for training
torch.save((input_ids, attention_mask, labels), 'preprocessed_data.pt')

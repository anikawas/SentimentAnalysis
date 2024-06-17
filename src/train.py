import torch
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

def train_model():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load preprocessed data
    input_ids, attention_mask, labels = torch.load('preprocessed_data.pt')

    # Move tensors to the GPU if available
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # Create the dataset
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=32)

    # Initialize the BERT model for sequence classification and move it to the GPU if available
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=13)
    model.to(device)

    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Training loop
    for epoch in range(4):  # Number of epochs
        model.train()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids_batch = batch[0]
            attention_mask_batch = batch[1]
            labels_batch = batch[2]

            outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Log progress
            if step % 10 == 0 and step != 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # Save the trained model
    torch.save(model.state_dict(), '../models/bert_sentiment_model.pth')

if __name__ == "__main__":
    train_model()

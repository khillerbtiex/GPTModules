from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from entityllm_enhanced import (
    GPT2WithDivergenceConvergence,
    DivergenceConvergenceModule,
)
import pickle

# Load dataset
ds = load_dataset("anthracite-org/nopm_claude_writing_fixed")
train_texts = ds["train"]["conversations"]

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "openai-community/gpt2-large"
)  # Replace "gpt2" with your specific model

# Prepare a system message to start each conversation
system_message = {"from": "system", "value": "You are a helpful assistant."}

# Concatenate and format the dataset
formatted_texts = []
for conversation in train_texts:
    # Add the system message at the beginning of each conversation
    conversation_text = f"{system_message['from']}: {system_message['value']}\n"

    for message in conversation:
        # Format each message with appropriate tag and add it to the conversation text
        tag = "human" if message["from"] == "human" else "assistant"
        conversation_text += f"{tag}: {message['value']}\n"

    formatted_texts.append(conversation_text)

# Tokenize the dataset with a max sequence length of 512
tokenized_texts = tokenizer(
    formatted_texts,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

# The tokenized_texts can now be used to create PyTorch DataLoaders or fed directly into the GPT model for training
train_dataloader = DataLoader(tokenized_texts, batch_size=8)

model = GPT2WithDivergenceConvergence(model_name="openai-community/gpt2-large")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Define optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
num_epochs = 36
total_steps = len(train_dataloader) * num_epochs


# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
from tqdm import tqdm


def load_model(model, optimizer, path):
    # Load the checkpoint from the file using pickle
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    # Restore the model and optimizer state dictionaries
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore the epoch
    last_epoch = checkpoint["epoch"]

    print(
        f"Model and optimizer state loaded from {path}, resuming at last_epoch {last_epoch}"
    )

    return model, optimizer, last_epoch


# Function to save model, optimizer state, and current epoch
def save_model(model, optimizer, last_epoch, path="models/model_checkpoint.pkl"):
    # Save model and optimizer state dictionaries, and the current epoch
    checkpoint = {
        "epoch": last_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Model saved at epoch {last_epoch} to {path}")


# Load the model, optimizer, and epoch from the file
model, optimizer, last_epoch = load_model(
    model, optimizer, "models/model_checkpoint.pkl"
)
optimizer = optim.AdamW(model.parameters(), lr=1e-6)
for layer in model.gpt2.transformer.h[:24]:
    for param in layer.parameters():
        param.requires_grad = False


# Training loop
model.train()
for epoch in tqdm(range(6)):
    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"].to(model.gpt2.device)
        attention_mask = batch["attention_mask"].to(model.gpt2.device)

        # Shift labels to the left by one for next-token prediction
        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        masks = attention_mask[:, :-1].contiguous()

        # Forward pass
        outputs = model(input_ids=inputs, attention_mask=masks)
        logits = outputs  # [batch_size, seq_length, vocab_size]

        # Compute loss
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed. Average Loss: {loss.item():.4f}")


def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[:, -1, :]  # Get logits for the last token
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token_id)], dim=-1
            )

    generated_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    return generated_text


# Example usage
prompt = (
    "system: You are a helpful assistan.\nhuman: Write me a good story about an angry mouse.\nassistant:",
)
generated = generate_text(model, tokenizer, prompt, max_length=50)
print(generated)
save_model(model, optimizer, epoch, "models/model_checkpoint_2.pkl")

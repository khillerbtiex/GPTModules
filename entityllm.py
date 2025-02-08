import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model
from torch.nn import MultiheadAttention
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import pickle


class LegIdentificationSubmodule(nn.Module):
    def __init__(self, embed_dim, num_heads, expansion_factor=2):
        super(LegIdentificationSubmodule, self).__init__()
        self.multihead_attn = MultiheadAttention(embed_dim, num_heads)
        self.expansion_factor = expansion_factor
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        if mask is not None:
            # Invert mask for PyTorch attention (1 = masked, 0 = unmasked)
            mask = ~mask.bool()
        attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask=mask)
        expanded = attn_output.repeat(self.expansion_factor, 1, 1)
        expanded = self.linear(expanded)
        return expanded


class ContextualEmbeddingPooling(nn.Module):
    def __init__(self, embed_dim, pool_size):
        super(ContextualEmbeddingPooling, self).__init__()
        self.pool_size = pool_size
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Ensure we don't try to pool more than available
        pool_size = min(self.pool_size, x.size(0))
        pooled, _ = torch.topk(x, pool_size, dim=0)
        pooled = self.linear(pooled)
        return pooled


class ExplorationHead(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ExplorationHead, self).__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = ~mask.bool()
        return self.attn(x, x, x, key_padding_mask=mask)[0]


class DivergenceConvergenceModule(nn.Module):
    def __init__(
        self, embed_dim, num_heads, pool_size=10, relationship_types=5, top_k=5
    ):
        super(DivergenceConvergenceModule, self).__init__()
        self.leg_id = LegIdentificationSubmodule(embed_dim, num_heads)
        self.pooling = ContextualEmbeddingPooling(embed_dim, pool_size)
        self.exploration = ExplorationHead(embed_dim, num_heads)
        self.final_linear = nn.Linear(embed_dim * 2, embed_dim)

        # Initialize device attribute
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return super(DivergenceConvergenceModule, self).to(device)

    def forward(self, x, mask=None):
        # Ensure inputs are on the correct device
        x = x.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Phase 1: Process sequence
        legs = self.leg_id(x, mask)
        pooled_legs = self.pooling(legs)
        explored = self.exploration(pooled_legs, mask)

        # Repeat explored features to match input sequence length
        explored_mean = explored.mean(dim=0, keepdim=True)  # (1, batch_size, embed_dim)
        explored_repeated = explored_mean.repeat(
            x.size(0), 1, 1
        )  # (seq_length, batch_size, embed_dim)

        # Combine with original input
        combined = torch.cat(
            [x, explored_repeated], dim=-1
        )  # (seq_length, batch_size, embed_dim*2)
        output = self.final_linear(combined)  # (seq_length, batch_size, embed_dim)

        return output


class GPT2WithDivergenceConvergence(nn.Module):
    def __init__(self, divergence_convergence_module):
        super(GPT2WithDivergenceConvergence, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained("openai-community/gpt2-large")
        # Freeze all GPT-2 parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False
        self.divergence_convergence = divergence_convergence_module
        self.lm_head = nn.Linear(
            self.gpt2.config.n_embd, self.gpt2.config.vocab_size, bias=False
        )

        # Initialize device attribute
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        self.gpt2 = self.gpt2.to(device)
        self.divergence_convergence = self.divergence_convergence.to(device)
        self.lm_head = self.lm_head.to(device)
        return super(GPT2WithDivergenceConvergence, self).to(device)

    def forward(self, input_ids, attention_mask=None):
        # Ensure inputs are on the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # GPT-2 processing
        outputs = self.gpt2(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        hidden_states = outputs.last_hidden_state.transpose(0, 1)

        # Apply Divergence-Convergence Module
        dc_output = self.divergence_convergence(hidden_states, mask=attention_mask)
        dc_output = dc_output.transpose(0, 1)

        # Generate logits
        logits = self.lm_head(dc_output)

        return logits


def train_model(
    model,
    train_dataloader,
    epochs,
    learning_rate,
    device,
    test_sentence=None,
    tokenizer=None,
):
    if test_sentence is not None:
        test_input_ids = tokenizer(test_sentence, return_tensors="pt").input_ids

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs // 2,
        eta_min=1e-6,
    )
    model = model.to(device)
    model.train()

    progress_bar = tqdm(range(epochs), desc="Training")

    for epoch in progress_bar:
        total_loss = 0
        batch_count = 0

        for batch in tqdm(train_dataloader):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(input_ids)

            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            batch_count += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": total_loss / batch_count})
        if test_input_ids is not None:
            model.eval()  # Set model to evaluation mode
            # Shape: [1, seq_length]

            with torch.no_grad():
                # Forward pass
                outputs = model(test_input_ids)
                # outputs shape: [1, seq_length, vocab_size]

                # Get the logits for the last position
                last_token_logits = outputs[0, -1, :]  # Shape: [vocab_size]

                # Optionally, apply softmax to get probabilities
                probabilities = torch.softmax(last_token_logits, dim=-1)

                # Get the top 5 predicted tokens and their probabilities
                top_k = 5
                top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

                # Decode the predicted tokens
                predicted_tokens = tokenizer.convert_ids_to_tokens(
                    top_k_indices.cpu().numpy()
                )

                # Print the results
                print(f"\nTest Sentence: '{test_sentence}'")
                print("Top predictions for the next token:")
                for i in range(top_k):
                    token = predicted_tokens[i]
                    prob = top_k_probs[i].item()
                    print(f"  {i+1}: {token} (probability: {prob:.4f})")
            model.train()  # Set model back to training mode
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
        scheduler.step()
    save_model(model, optimizer, 0, "models/model_checkpoint.pkl")


def prepare_data(dataset_name, batch_size, tokenizer):
    # Load dataset
    dataset = load_dataset(dataset_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    def collate_fn(examples):
        input_ids = torch.tensor([example["input_ids"][:-1] for example in examples])
        labels = torch.tensor([example["input_ids"][1:] for example in examples])
        return input_ids, labels

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_dataloader


def load_model(model, optimizer, path):
    # Load the checkpoint from the file using pickle
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    # Restore the model and optimizer state dictionaries
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore the epoch
    epoch = checkpoint["epoch"]

    print(f"Model and optimizer state loaded from {path}, resuming at epoch {epoch}")

    return model.to("cuda"), optimizer, epoch


def save_model(model, optimizer, epoch, path="models/model_checkpoint.pkl"):
    # Save model and optimizer state dictionaries, and the current epoch
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Model saved at epoch {epoch} to {path}")


# Usage example
if __name__ == "__main__":

    # Model configuration
    embed_dim = 1280  # GPT-2 base
    num_heads = 20
    pool_size = 10
    relationship_types = 5
    top_k = 5

    # Training configuration
    batch_size = 8
    epochs = 32
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    divergence_convergence = DivergenceConvergenceModule(
        embed_dim, num_heads, pool_size, relationship_types, top_k
    )

    model = GPT2WithDivergenceConvergence(divergence_convergence)
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    # Prepare data
    train_dataloader = prepare_data(
        "Lambent/creative-writing-2048-fineweb-edu-sample", batch_size, tokenizer
    )
    test_sentence = "He walked down the alley, scraping his feet on the "
    # Train model
    train_model(
        model, train_dataloader, epochs, learning_rate, device, test_sentence, tokenizer
    )

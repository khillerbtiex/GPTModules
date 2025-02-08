from datasets import load_dataset

# Load WikiText dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Example of splitting the dataset
train_texts = dataset["train"]["text"]
validation_texts = dataset["validation"]["text"]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from transformers import AutoTokenizer

# Load a GPT-2 tokenizer (handles tokenization and sub-tokenization)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True
)
tokenizer.pad_token = tokenizer.unk_token  # Set the pad token to be the unk token
tokenizer.pad_token_id = (
    tokenizer.unk_token_id
)  # Set the pad token ID to be the unk token ID
tokenizer.padding_side = "left"  # Set the padding side to be the left side


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HypergraphTokenTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=1600,
        nhead=32,
        num_layers=48,
        dropout=0.1,
        hyperedge_size=3,
        num_hyperedges=64,
    ):
        super().__init__()
        self.d_model = d_model
        self.hyperedge_size = hyperedge_size
        self.num_hyperedges = num_hyperedges

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # Separate heads for different aspects
        self.semantic_heads = nhead // 4
        self.composition_heads = nhead // 4
        self.reasoning_heads = nhead // 4
        self.research_heads = nhead // 4

        # Create hypergraph transformer layers
        self.layers = nn.ModuleList(
            [
                HypergraphDecompositionLayer(
                    d_model, nhead, dropout, hyperedge_size, num_hyperedges
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, src, attention_mask=None):
        # Initial embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Process through hypergraph transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.output_projection(x)


class HypergraphDecompositionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.1, hyperedge_size=3, num_hyperedges=64
    ):
        super().__init__()

        # Specialized hypergraph attention mechanisms
        self.semantic_attention = HypergraphAttention(
            d_model, nhead // 4, dropout, hyperedge_size, num_hyperedges
        )
        self.composition_attention = HypergraphAttention(
            d_model, nhead // 4, dropout, hyperedge_size, num_hyperedges
        )
        self.reasoning_attention = HypergraphAttention(
            d_model, nhead // 4, dropout, hyperedge_size, num_hyperedges
        )
        self.research_attention = HypergraphAttention(
            d_model, nhead // 4, dropout, hyperedge_size, num_hyperedges
        )

        # Hyperedge importance learning
        self.edge_importance = nn.Parameter(torch.ones(num_hyperedges))
        self.softmax = nn.Softmax(dim=0)

        # Feed-forward networks with hypergraph-aware processing
        self.semantic_ff = HypergraphFeedForward(d_model, dropout)
        self.composition_ff = HypergraphFeedForward(d_model, dropout)
        self.reasoning_ff = HypergraphFeedForward(d_model, dropout)
        self.research_ff = HypergraphFeedForward(d_model, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        # Get normalized edge importance weights
        edge_weights = self.softmax(self.edge_importance)

        # Apply specialized hypergraph attention
        semantic = self.semantic_attention(x, x, x, attention_mask, edge_weights)
        composition = self.composition_attention(x, x, x, attention_mask, edge_weights)
        reasoning = self.reasoning_attention(x, x, x, attention_mask, edge_weights)
        research = self.research_attention(x, x, x, attention_mask, edge_weights)

        # Combine aspects with learned weights
        combined = (semantic + composition + reasoning + research) / 4
        x = self.norm1(x + combined)

        # Hypergraph-aware feed-forward processing
        semantic_ff = self.semantic_ff(x, edge_weights)
        composition_ff = self.composition_ff(x, edge_weights)
        reasoning_ff = self.reasoning_ff(x, edge_weights)
        research_ff = self.research_ff(x, edge_weights)

        # Combine feed-forward outputs
        combined_ff = (semantic_ff + composition_ff + reasoning_ff + research_ff) / 4
        x = self.norm2(x + combined_ff)

        return x


class HypergraphAttention(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.1, hyperedge_size=3, num_hyperedges=64
    ):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.hyperedge_size = hyperedge_size
        self.num_hyperedges = num_hyperedges

        # Projections for queries, keys, values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Hyperedge projections
        self.hyperedge_proj = nn.Linear(d_model * hyperedge_size, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask=None, edge_weights=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Project and reshape for attention
        q = (
            self.q_proj(query)
            .view(batch_size, -1, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, -1, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, -1, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

        # Generate hyperedge connections
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Create hyperedge attention patterns
        hyperedge_attn = self._create_hyperedge_attention(
            scores.size(), seq_len, scores.device
        )

        # Apply edge weights if provided
        if edge_weights is not None:
            # Reshape edge_weights to match hyperedge dimensions: [1, 1, 1, num_hyperedges]
            edge_weights = edge_weights.view(1, 1, 1, -1)

            # Create a mapping matrix from num_hyperedges to seq_len
            # Shape: [1, 1, num_hyperedges, seq_len]
            mapping_matrix = torch.zeros(
                1, 1, self.num_hyperedges, seq_len, device=edge_weights.device
            )
            positions_per_edge = seq_len // self.num_hyperedges
            for i in range(self.num_hyperedges):
                start_idx = i * positions_per_edge
                end_idx = (
                    start_idx + positions_per_edge
                    if i < self.num_hyperedges - 1
                    else seq_len
                )
                mapping_matrix[0, 0, i, start_idx:end_idx] = 1.0

            # Apply edge weights to mapping matrix
            # Shape: [1, 1, num_hyperedges, seq_len] * [1, 1, 1, num_hyperedges] -> [1, 1, seq_len, seq_len]
            weighted_mapping = torch.matmul(
                mapping_matrix.transpose(-2, -1), edge_weights.transpose(-2, -1)
            )

            # Expand to match batch and head dimensions
            weighted_mapping = weighted_mapping.expand(
                batch_size, self.nhead, seq_len, seq_len
            )

            # Apply weighted mapping to hyperedge attention
            hyperedge_attn = hyperedge_attn * weighted_mapping

        # Combine with original attention
        attn = F.softmax(scores + hyperedge_attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(out)

    def _create_hyperedge_attention(self, scores_size, seq_len, device):
        batch_size, num_heads, tgt_len, src_len = scores_size

        # Initialize hyperedge attention tensor
        hyperedge_attn = torch.zeros(
            (batch_size, num_heads, tgt_len, src_len), device=device
        )

        # Generate hyperedge patterns that match sequence length
        for i in range(self.num_hyperedges):
            # Create random hyperedge connections
            for j in range(tgt_len):
                # Generate random connections for each target position
                indices = torch.randint(
                    0, src_len, (self.hyperedge_size,), device=device
                )
                mask = torch.zeros(src_len, device=device)
                mask[indices] = 1.0 / self.hyperedge_size
                hyperedge_attn[:, :, j, :] += mask

        # Normalize hyperedge attention
        hyperedge_attn = hyperedge_attn / self.num_hyperedges

        return hyperedge_attn


class HypergraphFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Hypergraph-aware processing
        self.hypergraph_gate = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_weights=None):
        # Standard feed-forward
        h = self.ff1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.ff2(h)

        # Hypergraph-aware gating
        if edge_weights is not None:
            gate = torch.sigmoid(self.hypergraph_gate(x))
            h = h * gate

        return self.layer_norm(h)


# Example training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = HypergraphTokenTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=1024,
    nhead=16,
    num_layers=16,
    dropout=0.1,
    hyperedge_size=3,
    num_hyperedges=16,
)
max_length = 512  # You can adjust this based on your model's input size


def prepare_source_target(examples):
    """
    Create source and target sequences from the text.
    Source is the original text, target is shifted by one position.
    """
    texts = examples["text"]
    sources = []
    targets = []

    for text in texts:
        # Create source-target pairs
        sources.append(text)
        # For target, we'll let the tokenizer handle the shifting later
        targets.append(text)

    return {"source": sources, "target": targets}


def tokenize_and_align(examples, max_length=512):
    """
    Tokenize both source and target texts and align them for next-token prediction.
    """
    # Tokenize source sequences
    source_encoding = tokenizer(
        examples["source"],
        truncation=True,
        max_length=max_length,
        padding="max_length",  # Always pad to max_length
        return_tensors="pt",
    )

    # Tokenize target sequences
    target_encoding = tokenizer(
        examples["target"],
        truncation=True,
        max_length=max_length,
        padding="max_length",  # Always pad to max_length
        return_tensors="pt",
    )

    input_ids = source_encoding["input_ids"]
    attention_mask = source_encoding["attention_mask"]

    # Create shifted labels from target_encoding
    labels = target_encoding["input_ids"].clone()

    # Shift labels to the left by 1 and add -100 as the last token
    labels = labels[:, 1:]  # Remove first token
    labels = torch.cat(
        [labels, torch.full((labels.shape[0], 1), -100)], dim=1
    )  # Add -100 at the end

    # Apply mask to labels - set padded positions to -100
    labels = labels.masked_fill(attention_mask == 0, -100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# First, create source-target pairs
dataset_with_pairs = dataset.map(
    prepare_source_target, batched=True, remove_columns=dataset["train"].column_names
)

# Then tokenize and create aligned sequences
tokenized_dataset = dataset_with_pairs.map(
    tokenize_and_align, batched=True, remove_columns=["source", "target"]
)

# Split into train and validation
tokenized_train = tokenized_dataset["train"]
tokenized_val = tokenized_dataset["validation"]
from torch.utils.data import DataLoader

# Define batch size
batch_size = 16  # Adjust according to your available GPU memory

# Convert the datasets to PyTorch format
train_dataset = tokenized_train.with_format("torch")
val_dataset = tokenized_val.with_format("torch")

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()
model.train()
model = model.to(device)
from tqdm import tqdm

for epoch in tqdm(range(10)):

    for batch in tqdm(
        train_dataloader
    ):  # Assume you have a DataLoader setup for WikiText
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Shift the input for next-token prediction
        logits = model(input_ids)

        # Reshape for cross-entropy
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")


def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids)

            # Compute the loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            # Update total loss and total number of tokens (ignoring padding)
            total_loss += loss.item() * labels.size(0)  # Multiply by batch size
            total_tokens += (
                (labels != -100).sum().item()
            )  # Count valid tokens (not masked with -100)

    # Compute average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()


# Example usage:

# Initialize the loss function (CrossEntropyLoss)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens (-100)

# Call the evaluate function after each epoch (assuming `val_dataloader` and `model` are defined)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_loss, val_perplexity = evaluate(model, val_dataloader, criterion, device)

print(f"Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")

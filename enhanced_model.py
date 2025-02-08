import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import os
import re
import pickle

import opt_einsum as oe


class ImprovedRandomProjection(nn.Module):
    def __init__(self, input_dim, proj_dim, num_projections=8):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.num_projections = num_projections

        # Use multiple random matrices for better feature capture
        self.projections = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(input_dim, proj_dim // num_projections)
                    / math.sqrt(input_dim),
                    requires_grad=False,
                )
                for _ in range(num_projections)
            ]
        )

        # Learned scaling factors for each projection
        self.scales = nn.Parameter(torch.ones(num_projections))

    def forward(self, x):
        # Project using multiple matrices in parallel
        projections = [F.linear(x, proj) for proj in self.projections]
        # Scale and concatenate
        scaled = [proj * scale for proj, scale in zip(projections, self.scales)]
        return torch.cat(scaled, dim=-1)


class FastNystromAttention(nn.Module):
    def __init__(
        self, block_size, num_heads, d_model, num_landmarks=64, sample_ratio=0.25
    ):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_landmarks = num_landmarks
        self.sample_ratio = sample_ratio
        self.scale = 1 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)  # Combined QKV projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Add relative positional embeddings
        self.max_seq_len = 2048  # Adjust based on your needs
        self.rel_pos_emb = nn.Parameter(
            torch.randn(2 * self.max_seq_len - 1, self.head_dim)
            / math.sqrt(self.head_dim)
        )

    def get_rel_pos_emb(self, seq_len):
        # Generate relative position indices
        pos_indices = torch.arange(seq_len, device=self.rel_pos_emb.device)
        rel_pos_indices = pos_indices.unsqueeze(1) - pos_indices.unsqueeze(0)
        rel_pos_indices += self.max_seq_len - 1
        return self.rel_pos_emb[rel_pos_indices]

    def fast_attention(self, q, k, v, rel_pos, mask=None):
        B, H, L, D = q.shape

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add relative positional bias
        attn = attn + rel_pos

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # Apply softmax
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        return out

    def nystrom_attention(self, q, k, v, rel_pos, mask=None):
        B, H, L, D = q.shape

        # Randomly sample landmarks
        if L > self.num_landmarks:
            indices = torch.randperm(L, device=q.device)[: self.num_landmarks]
            k_landmarks = k[:, :, indices, :]
            v_landmarks = v[:, :, indices, :]
        else:
            k_landmarks = k
            v_landmarks = v

        # Compute Nyström components
        kernel_1 = torch.matmul(q, k_landmarks.transpose(-2, -1)) * self.scale
        kernel_2 = torch.matmul(k_landmarks, k_landmarks.transpose(-2, -1)) * self.scale
        kernel_3 = torch.matmul(k, k_landmarks.transpose(-2, -1)) * self.scale

        # Add relative positional bias
        kernel_1 = kernel_1 + rel_pos[:, :, :, : self.num_landmarks]

        if mask is not None:
            if L > self.num_landmarks:
                landmark_mask = mask[:, :, indices]
            else:
                landmark_mask = mask
            kernel_1 = kernel_1.masked_fill(landmark_mask == 0, float("-inf"))

        # Moore-Penrose pseudoinverse
        kernel_2 = (
            kernel_2 + torch.eye(kernel_2.size(-1), device=kernel_2.device) * 1e-6
        )
        kernel_2_inv = torch.linalg.pinv(kernel_2)

        # Compute attention
        attn = torch.matmul(torch.matmul(kernel_1, kernel_2_inv), kernel_3)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return out

    def forward(self, x, mask=None):
        B, N, C = x.shape

        # Combined QKV projection
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Get relative positional embeddings
        rel_pos = self.get_rel_pos_emb(N)

        # Use Nyström attention for long sequences, regular attention for short ones
        if N > self.num_landmarks * 2:
            out = self.nystrom_attention(q, k, v, rel_pos, mask)
        else:
            out = self.fast_attention(q, k, v, rel_pos, mask)

        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class ImprovedSparsePerceiver(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        input_dim,
        proj_dim,
        latent_dim,
        num_latents,
        num_heads,
        block_size,
        num_layers=1,
        num_landmarks=64,
        sample_ratio=0.25,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Improved random projection
        self.random_proj = ImprovedRandomProjection(embedding_dim, proj_dim)

        # Learned latent tokens with improved initialization
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)

        # Layer stack with improved attention
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": FastNystromAttention(
                            block_size, num_heads, proj_dim, num_landmarks, sample_ratio
                        ),
                        "norm1": nn.LayerNorm(latent_dim),
                        "norm2": nn.LayerNorm(latent_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(latent_dim, 4 * latent_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(4 * latent_dim, latent_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Improved output projection with layer normalization
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(num_latents * latent_dim, 4 * vocab_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * vocab_size, vocab_size),
        )

        # Initialize weights with improved method
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.02, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, mask=None):
        B = x.size(0)

        # Embedding with dropout
        x = self.dropout(self.embedding(x))

        # Project input
        proj_x = self.random_proj(x)

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Process through layers
        for layer in self.layers:
            # Self-attention with residual
            residual = latents
            latents = layer["norm1"](latents)
            latents = layer["attn"](latents, mask) + residual

            # FFN with residual
            residual = latents
            latents = layer["norm2"](latents)
            latents = layer["ffn"](latents) + residual

        # Final normalization and projection
        latents = self.output_norm(latents)
        outputs = self.output_layer(latents.reshape(B, -1))

        return outputs


# Improved training function
def train_step(model, batch, optimizer, scaler, device):
    model.train()
    inputs, targets = [x.to(device) for x in batch]

    with torch.autocast():
        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

    # Gradient scaling and backward pass
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()

    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    return loss.item()


def count_parameters(model):
    def format_number(num):
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        return str(num)

    total_params = 0
    details = {}

    # Random Projection
    rand_proj_params = model.random_proj.proj.numel()
    details["random_projection"] = rand_proj_params
    total_params += rand_proj_params

    # Learnable latents
    latents_params = model.latents.numel()
    details["learnable_latents"] = latents_params
    total_params += latents_params

    # Self-attention layers
    layer_details = {}
    for i, layer in enumerate(model.self_attn_layers):
        layer_params = {}

        # Attention parameters
        attn_params = sum(p.numel() for p in layer["attn"].parameters())
        layer_params["attention"] = attn_params

        # Projection parameters
        proj_to_latent_params = sum(
            p.numel() for p in layer["proj_to_latent"].parameters()
        )
        latent_to_proj_params = sum(
            p.numel() for p in layer["latent_to_proj"].parameters()
        )
        layer_params["projections"] = proj_to_latent_params + latent_to_proj_params

        # Layer norms
        ln_params = sum(p.numel() for p in layer["layer_norm1"].parameters()) + sum(
            p.numel() for p in layer["layer_norm2"].parameters()
        )
        layer_params["layer_norms"] = ln_params

        # FFN parameters
        ffn_params = sum(p.numel() for p in layer["ffn"].parameters())
        layer_params["ffn"] = ffn_params

        layer_total = sum(layer_params.values())
        layer_params["total"] = layer_total
        layer_details[f"layer_{i}"] = layer_params
        total_params += layer_total

    # Output layer
    output_params = sum(p.numel() for p in model.output_layer.parameters())
    details["output_layer"] = output_params
    total_params += output_params

    # Prepare the detailed report
    report = {
        "total_parameters": total_params,
        "total_parameters_formatted": format_number(total_params),
        "component_details": {
            "random_projection": format_number(details["random_projection"]),
            "learnable_latents": format_number(details["learnable_latents"]),
            "output_layer": format_number(details["output_layer"]),
            "layers": {
                k: {sk: format_number(sv) for sk, sv in v.items()}
                for k, v in layer_details.items()
            },
        },
        "raw_numbers": {
            "random_projection": details["random_projection"],
            "learnable_latents": details["learnable_latents"],
            "output_layer": details["output_layer"],
            "layers": layer_details,
        },
    }

    return report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"
# Initialize the model
vocab_size = tokenizer.vocab_size  # Typically 50257 for GPT-2
embedding_dim = 768
input_dim = 1024
enhanced_model = ImprovedSparsePerceiver(
    vocab_size=50257,  # GPT-2 vocabulary size
    embedding_dim=768,
    input_dim=1024,
    proj_dim=512,
    latent_dim=512,
    num_latents=64,
    num_heads=16,
    block_size=32,
    num_layers=8,
    dropout=0.1,
).half()
scaler = torch.GradScaler()
print(
    f"Initialized Perciever Model of {count_parameters(enhanced_model)["total_parameters_formatted"]} parameters"
)

optimizer = torch.optim.AdamW(enhanced_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=500, eta_min=1e-6
)
# Expected output: [32, num_latents, latent_dim]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train function
def train(model, optimizer, train_loader):
    criterion = nn.CrossEntropyLoss()  # Assuming diclassification
    model.train()

    total_loss = 0.0
    for batch in tqdm(train_loader):
        try:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Runs the forward pass with autocasting.
            with torch.autocast(device_type="cuda", enabled=True):
                outputs = model(inputs)
                print(f"Outputs: {outputs}")
                print(f"Targets: {targets}")
                outputs = outputs.view(-1, vocab_size)
                targets = targets.view(-1)
                loss = criterion(outputs, targets)

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaled_loss = scaler.scale(loss)
            print(loss)
            scaled_loss.backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.

            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            total_loss += scaled_loss.detach().item()
            print(total_loss)
        except Exception as e:
            print(e)

    avg_loss = total_loss / len(train_loader)

    return avg_loss


# Test function
def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            if len(targets.shape) > 1:
                targets = targets.squeeze()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss/len(test_loader)}")


from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset(
    "wikitext", "wikitext-103-v1", split="train[:05%]"
).train_test_split(test_size=0.1, shuffle=True)["train"]


# Tokenize without padding
def tokenize_no_padding(examples):
    return tokenizer(
        examples["text"], padding=False, truncation=True, max_length=input_dim
    )


# Tokenize dataset
print("Tokenizing dataset without padding...")
tokenized_dataset = dataset.map(tokenize_no_padding, batched=True)

# Get input lengths (sequence lengths)
seq_lengths = [len(seq) for seq in tokenized_dataset["input_ids"]]

# Prepare input and target tensors
new_inputs = []
new_targets = []
max_length = input_dim
print("Randomly sampling inputs and targets based on sequence length...")

for i, seq in enumerate(tokenized_dataset["input_ids"]):
    seq_length = seq_lengths[i]
    start_index = seq_length // 8
    if seq_length > 3:  # If sequence is too long, randomly select a smaller subset
        random_index = random.randint(
            max(2, start_index), min(max_length - 2, seq_length - 2)
        )
        if random_index > 0:  # Ensure we sample at least one token
            new_inputs.append(torch.tensor(seq[:random_index]))
            new_targets.append(torch.tensor(seq[random_index]))
    else:
        continue

# Pad sequences to the max length

print(f"Padding sequences to length {max_length}...")


def pad_to_max_length(
    sequence: torch.Tensor, max_length: int, pad_value: int
) -> torch.Tensor:
    print(sequence, type(sequence))
    if sequence.size(0) < max_length - 1:
        # Calculate padding length
        pad_len = max_length - sequence.size(0) - 1
        # Pad the sequence
        return F.pad(sequence, (0, pad_len), value=pad_value)
    return sequence


valid_pairs = []

# Validate and collect valid pairs
for input_seq, target_seq in zip(new_inputs, new_targets):
    try:
        # Check if sequences have dimensions
        _ = input_seq.size(0)

        # Pad sequences if they're valid
        padded_input = pad_to_max_length(input_seq, max_length, tokenizer.pad_token_id)

        valid_pairs.append((padded_input, target_seq))

    except (IndexError, RuntimeError):
        # Skip pairs where either sequence has no dimensions
        continue
padded_inputs, padded_targets = zip(*valid_pairs)
padded_inputs = torch.stack([seq.long() for seq in padded_inputs])
padded_targets = torch.stack([seq.long() for seq in padded_targets])

train_dataset = TensorDataset(padded_inputs, padded_targets)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
)
print(train_dataset[0])


def find_latest_checkpoint(
    checkpoint_dir="models", pattern=r"model_checkpoint_(\d+)\.pth"
):
    # Regex pattern to extract epoch number from filename
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(pattern, f)]
    if not checkpoint_files:
        return None, 0  # No checkpoints found, start from epoch 0

    checkpoints = [int(f.split(".")[0].split("_")[-1]) for f in checkpoint_files]
    # Find the file with the maximum epoch
    latest_epoch = max(checkpoints)
    latest_checkpoint = f"models/model_checkpoint_{latest_epoch}.pth"

    return latest_checkpoint, latest_epoch


# Function to load model and optimizer state from the latest checkpoint
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

    return model, optimizer, epoch


# Function to save model, optimizer state, and current epoch
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


checkpoint_path, start_epoch = find_latest_checkpoint()
if checkpoint_path:
    enhanced_model, optimizer, start_epoch = load_model(
        enhanced_model, optimizer, checkpoint_path
    )
else:
    start_epoch = 0  # Start from scratch if no checkpoint exists
    # init_weights_from_gpt2(
    #     enhanced_model,
    # )
# Define the total number of epochs
total_epochs = 500

# Calculate the remaining epochs to train
remaining_epochs = total_epochs - start_epoch
enhanced_model = enhanced_model.to(device)
if remaining_epochs > 1:
    print("Training....")
    total_avg_loss = []
    # Train the model starting from the loaded epoch
    for epoch in tqdm(range(start_epoch + 1, total_epochs)):

        epoch_avg_loss = train_step(enhanced_model, optimizer, train_loader)
        total_avg_loss += [epoch_avg_loss]
        print(f"Epoch {epoch+1}/{total_epochs}, Loss: {np.mean(total_avg_loss):.4f}")

        # Save the model after every epoch
        save_model(
            enhanced_model,
            optimizer,
            epoch,
            path=f"models/model_checkpoint_{epoch}.pth",
        )
        scheduler.step()
else:
    print(f"Training already completed for {total_epochs-1} epochs.")

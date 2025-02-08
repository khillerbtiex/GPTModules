import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import get_scheduler
import numpy as np
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from grokfast import gradfilter_ema
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import random
from torch.quantization import quantize_dynamic
from torch.optim.lr_scheduler import _LRScheduler


class SmoothCosineAnnealingRestarts(_LRScheduler):
    """
    Cosine annealing scheduler with warm restarts where each restart's peak LR
    is lower than the previous one, creating a smoother overall decay pattern.
    """

    def __init__(
        self, optimizer, T_0, T_mult=2, eta_min=0, peak_decay=0.5, last_epoch=-1
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            T_0: Initial period of cosine decay (in steps)
            T_mult: Factor by which period length increases after each restart
            eta_min: Minimum learning rate
            peak_decay: Factor by which peak learning rate decreases at each restart
            last_epoch: The index of last epoch
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.peak_decay = peak_decay
        self.T_cur = 0
        self.T_i = T_0  # Current period
        self.peaks = [
            group["lr"] for group in optimizer.param_groups
        ]  # Initial peaks for each param group
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Calculate progress within current cycle
        progress = self.T_cur / self.T_i

        # Cosine decay for current cycle
        cos_decay = (1 + math.cos(math.pi * progress)) / 2

        # Calculate learning rates
        lrs = [self.eta_min + (peak - self.eta_min) * cos_decay for peak in self.peaks]

        # Update internal state if cycle is complete
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
            # Reduce peak learning rates for next cycle
            self.peaks = [peak * self.peak_decay for peak in self.peaks]

        return lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return self.layer_norm(residual + x)


class DynamicDomainAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.domain_dim = config.domain_dim

        self.domain_proj = nn.Sequential(
            LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.domain_dim),
        )

        self.domain_embeddings = nn.Parameter(
            torch.randn(config.num_domains, config.domain_dim)
            / (config.domain_dim**0.5)
        )

        # Contextual embeddings for unseen domains
        self.dynamic_proj = nn.Linear(config.hidden_size, config.num_domains)
        self.adapter = nn.Sequential(
            LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.adapter_dim),
            nn.SiLU(),
            nn.Linear(config.adapter_dim, config.hidden_size),
            LayerNorm(config.hidden_size),
        )

        self.adapter_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Static domain adaptation
        domain_query = self.domain_proj(x)
        static_scores = torch.matmul(domain_query, self.domain_embeddings.T)

        # Dynamic adaptation - now directly outputs scores for each domain
        dynamic_scores = self.dynamic_proj(x)

        # Combine static and dynamic scores
        domain_scores = static_scores + dynamic_scores
        domain_probs = F.softmax(domain_scores, dim=-1)

        # Apply adapter
        adapter_output = self.adapter(x)
        output = x + self.adapter_scale * adapter_output

        return output, domain_probs


class HierarchicalDomainAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Hierarchical layers
        self.use_hierarchical_attention = config.use_hierarchical_attention
        self.hierarchical_proj = None
        if self.use_hierarchical_attention:
            self.hierarchical_proj = nn.ModuleList(
                [
                    nn.Linear(self.head_dim, self.head_dim)
                    for _ in range(config.hierarchical_projs)
                ]
            )

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = LayerNorm(config.hidden_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.size()

        # Initial projections
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_hierarchical_attention:
            # Hierarchical processing
            original_shape = k.shape
            # Flatten the tensor while preserving the batch and the other dimensions (24, 16)
            k = k.reshape(-1, self.head_dim)
            v = v.reshape(-1, self.head_dim)

            for proj in self.hierarchical_proj:
                # Pass through the linear layer
                k = proj(k)
                v = proj(v)  # shape becomes [batch_size*24*16, 512]

            # Reshape back to the original tensor shape [24, 16, 512, 32]
            k = k.reshape(*original_shape)
            v = v.reshape(*original_shape)

        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask.unsqueeze(1).unsqueeze(2) if mask is not None else None,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)

        return self.layer_norm(x + attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.domain_adapter = DynamicDomainAdapter(config)
        self.attention = HierarchicalDomainAttention(config)
        self.mlp = MLP(config)
        self.prenorm = LayerNorm(config.hidden_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.prenorm(x)

        # Dynamic domain adaptation
        adapted_x, domain_probs = self.domain_adapter(x)

        # Hierarchical attention
        attn_output = self.attention(adapted_x, mask)

        # MLP
        output = self.mlp(attn_output)

        return output + residual, domain_probs


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding with scaled initialization
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.token_emb.weight, std=0.05)

        # Learnable position embeddings
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.max_seq_length, config.hidden_size)
        )
        nn.init.normal_(self.pos_emb, std=0.05)

        self.dropout = nn.Dropout(config.dropout)

        # Create transformer blocks
        block_list = []
        for _ in range(config.num_layers):
            config.use_hierarchical_attention = (
                True if _ < config.hierarchical_attention_layers else False
            )
            block_con = config
            block_list.append(TransformerBlock(config))
        self.blocks = nn.ModuleList(block_list)

        self.ln_final = LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.output.weight, std=0.05)

    def forward(self, x, mask=None):
        B, L = x.size()

        # Combine token and position embeddings
        pos_emb = self.pos_emb[:, :L, :]
        x = self.token_emb(x) + pos_emb
        x = self.dropout(x)

        # Process through transformer blocks
        domain_probs_all = []
        for block in self.blocks:
            x, domain_probs = block(x, mask)
            domain_probs_all.append(domain_probs)

        x = self.ln_final(x)
        logits = self.output(x)

        return logits, domain_probs_all


class GPTConfig:

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_length: int = 1024,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        domain_dim: int = 384,
        num_domains: int = 64,
        adapter_dim: int = 384,
        init_std: float = 0.05,
        use_dynamic_domain_adapter: bool = True,  # Enable Dynamic Domain Adapter
        use_hierarchical_attention: bool = True,  # Enable Hierarchical Domain Attention
        hierarchical_attention_layers: int = 4,  # Number of layers for hierarchical attention,
        hierarchical_projs: int = 2,  # Number of hierarchical projections
    ):
        assert (
            hidden_size % num_heads == 0
        ), "Hidden size must be divisible by number of heads"
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_domains = num_domains
        self.domain_dim = domain_dim
        self.adapter_dim = adapter_dim
        self.init_std = init_std
        self.use_dynamic_domain_adapter = use_dynamic_domain_adapter
        self.use_hierarchical_attention = use_hierarchical_attention
        self.hierarchical_attention_layers = hierarchical_attention_layers
        self.hierarchical_projs = hierarchical_projs


class FineWebDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512, ds_perc=0.14):
        # Load the FineWeb dataset
        percentage = int(ds_perc * 100)
        self.dataset = load_dataset(
            "deatos/fineweb-edu-10b-combined",
            split=split + f"[:{percentage}%]",
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve a text sequence
        item = self.dataset[idx]["text"]

        # Tokenize and truncate the sequence
        tokens = self.tokenizer(
            item,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )["input_ids"].squeeze(0)

        # Random truncation
        truncation_idx = random.randint(1, len(tokens))
        truncated_tokens = tokens[:truncation_idx]

        # Random left padding: calculate padding length
        pad_length = random.randint(0, self.max_length - len(truncated_tokens))

        # Add left padding (padding token is usually 50256 for GPT-2)
        padded_tokens = torch.cat(
            [torch.tensor([50256] * pad_length, dtype=torch.long), truncated_tokens]
        )

        # Add right padding
        token_space_left = self.max_length - len(padded_tokens)
        if token_space_left > 0:
            padded_tokens = torch.cat(
                [
                    padded_tokens,
                    torch.tensor([50256] * (token_space_left), dtype=torch.long),
                ]
            )

        # Ensure the sequence doesn't exceed max_length
        if len(padded_tokens) > self.max_length:
            padded_tokens = padded_tokens[: self.max_length]

        # Decode padded tokens back to text
        padded_text = self.tokenizer.decode(padded_tokens, skip_special_tokens=True)

        # Tokenize the padded text for the final input
        encodings = self.tokenizer(
            padded_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Create labels (shift input_ids left by 1)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Don't compute loss for last prediction

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.bool(),
            "labels": labels,
        }


class DollyDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512):
        self.dataset = load_dataset("databricks/databricks-dolly-15k", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Tokenize each component separately
        instruction_prefix = self.tokenizer("Instruction: ", add_special_tokens=False)[
            "input_ids"
        ]
        instruction_tokens = self.tokenizer(
            item["instruction"], add_special_tokens=False
        )["input_ids"]
        response_prefix = self.tokenizer("Response: ", add_special_tokens=False)[
            "input_ids"
        ]

        # Tokenize the truncated response
        response_tokens = self.tokenizer(
            item["response"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )["input_ids"].squeeze(0)

        # Random truncation of response
        truncation_idx = random.randint(1, len(response_tokens))
        truncated_response = response_tokens[:truncation_idx]

        # Convert all components to tensors
        instruction_prefix = torch.tensor(instruction_prefix, dtype=torch.long)
        instruction_tokens = torch.tensor(instruction_tokens, dtype=torch.long)
        response_prefix = torch.tensor(response_prefix, dtype=torch.long)

        # Calculate available space for padding
        total_content_length = (
            len(instruction_prefix)
            + len(instruction_tokens)
            + len(response_prefix)
            + len(truncated_response)
        )

        # Calculate max padding possible while leaving room for content
        max_padding_per_section = (self.max_length - total_content_length) // 3
        max_padding_per_section = max(0, max_padding_per_section)  # Ensure non-negative

        # Add random padding between sections
        pad1_length = random.randint(0, max_padding_per_section)
        pad2_length = random.randint(0, max_padding_per_section)
        pad3_length = random.randint(0, max_padding_per_section)

        # Create padding tensors
        pad1 = torch.tensor([50256] * pad1_length, dtype=torch.long)
        pad2 = torch.tensor([50256] * pad2_length, dtype=torch.long)
        pad3 = torch.tensor([50256] * pad3_length, dtype=torch.long)

        # Handle context if present
        if item["context"]:
            context_prefix = torch.tensor(
                self.tokenizer("Context: ", add_special_tokens=False)["input_ids"],
                dtype=torch.long,
            )
            context_tokens = torch.tensor(
                self.tokenizer(item["context"], add_special_tokens=False)["input_ids"],
                dtype=torch.long,
            )
            # Concatenate all components with padding
            tokens = torch.cat(
                [
                    instruction_prefix,
                    pad1,
                    instruction_tokens,
                    pad2,
                    context_prefix,
                    context_tokens,
                    pad3,
                    response_prefix,
                    truncated_response,
                ]
            )
        else:
            # Concatenate without context
            tokens = torch.cat(
                [
                    instruction_prefix,
                    pad1,
                    instruction_tokens,
                    pad2,
                    response_prefix,
                    pad3,
                    truncated_response,
                ]
            )

        # Add final padding if needed to reach max_length
        token_space_left = self.max_length - len(tokens)
        if token_space_left > 0:
            final_padding = torch.tensor([50256] * token_space_left, dtype=torch.long)
            tokens = torch.cat([tokens, final_padding])

        # Ensure we don't exceed max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(tokens)
        attention_mask[tokens == 50256] = 0

        # Create labels (shift input_ids left by 1)
        labels = tokens.clone()
        labels[:-1] = tokens[1:]
        labels[-1] = -100  # Don't compute loss for last prediction

        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask.bool(),
            "labels": labels,
        }


def train_model(
    model,
    tokenizer,
    train_dataloader,
    val_dataloader,
    optimizer=None,
    num_epochs=3,
    learning_rate=3e-4,
    eta_min=1e-6,
    max_grad_norm=1.0,
    warmup_steps=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Training on device: {device}")
    model = model.to(device)

    # Initialize optimizer and scheduler
    if not optimizer:
        optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Scheduler setup
    num_steps = num_epochs * len(train_dataloader)
    scheduler = SmoothCosineAnnealingRestarts(
        optimizer=optimizer,
        T_0=warmup_steps if warmup_steps else int(0.01 * num_steps),  # Warmup steps
        T_mult=2,  # Restart period multiplier (set as needed, e.g., 2 for doubling periods)
        eta_min=eta_min,
        peak_decay=0.9,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training statistics
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    grads = None
    with torch.autocast(device, dtype=torch.bfloat16):
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in train_pbar:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                logits, domain = model(input_ids, mask=attention_mask)

                # Calculate loss
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                # Backward pass
                loss.backward()
                grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update statistics
                total_train_loss += loss.item()
                train_pbar.set_postfix({"loss": loss.item()})

            # Validation
            model.eval()
            total_val_loss = 0
            val_pbar = tqdm(val_dataloader, desc="Validation")

            with torch.no_grad():
                for batch in val_pbar:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    logits, domain = model(input_ids, mask=attention_mask)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    )

                    total_val_loss += loss.item()
                    val_pbar.set_postfix(
                        {
                            "loss": loss.item(),
                        }
                    )

            # Calculate average losses
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_val_loss = total_val_loss / len(val_dataloader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            print(f"\nEpoch {epoch+1}:")
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Average validation loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                "best_model.pth",
            )
            print("Saved best model checkpoint")

    return (
        optimizer,
        train_losses,
        val_losses,
    )


import torch
import torch.nn.functional as F


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.7,
    top_k=50,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    model = model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs, doms = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature

            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)

            # Sample from the filtered distribution
            next_token = top_k_indices[0][torch.multinomial(probs[0], 1)]

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

            if next_token == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def split_ds(ds, train_perc=0.9, batch_size=16):
    # Calculate split sizes (90% train, 10% validation)
    total_size = len(ds)
    train_size = int(train_perc * total_size)
    val_size = total_size - train_size

    # Create splits
    train_dataset, val_dataset = random_split(
        ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4
    )
    return train_dataloader, val_dataloader


def main():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create config and model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=512,
        hidden_size=512,
        num_layers=32,
        num_heads=8,
        dropout=0.1,
        domain_dim=256,
        num_domains=4,
        adapter_dim=256,
        hierarchical_attention_layers=4,
    )
    model = GPT(config)
    model = torch.compile(model, backend="cudagraphs")

    model.load_state_dict(torch.load("best_model.pth")["model_state_dict"])
    """
    # Create full dataset
    full_dataset = FineWebDataset("train", tokenizer, ds_perc=0.05)

    # Train model
    train_dataloader, val_dataloader = split_ds(full_dataset, batch_size=24)

    optim, train_losses, val_losses = train_model(
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        num_epochs=1,
        learning_rate=5e-4,
        eta_min=5e-6,
    )

    full_dataset = DollyDataset("train", tokenizer=tokenizer)
    n = 8
    k = 4
    # Freeze first n layers
    for i, layer in enumerate(model.blocks[: n + k]):
        if i in range(n):
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.domain_adapter.parameters():
                param.requires_grad = True

    train_dataloader, val_dataloader = split_ds(full_dataset, batch_size=24)
    optim, ft_train_losses, ft_val_losses = train_model(
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        optimizer=None,
        num_epochs=8,
        learning_rate=5e-6,
        max_grad_norm=0.85,
    )
    torch.save(model.state_dict(), "gpt2_crosschop.pth")
    print("Saved model checkpoint")"""
    # Test generation
    test_prompt = (
        "Instruction: Explain what machine learning is to a 5-year old.\nResponse:"
    )
    generated_text = generate_text(
        model,
        tokenizer,
        test_prompt,
        max_length=100,
        temperature=0.7,
    )
    print("\nTest Generation:")
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")

    # Save Model


if __name__ == "__main__":
    main()

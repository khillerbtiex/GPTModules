import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from typing import Optional, Tuple, List, Dict, Any
import random
from peft import LoraConfig, get_peft_model, TaskType


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)


class AdaptiveSparseMLP(nn.Module):
    """Enhanced MLP with conditional sparsity and mixture of experts"""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts if hasattr(config, "num_experts") else 4
        self.hidden_size = config.hidden_size

        # Create multiple expert networks
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_size, 4 * config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(4 * config.hidden_size, config.hidden_size),
                )
                for _ in range(self.num_experts)
            ]
        )

        # Expert routing network
        self.router = nn.Sequential(
            LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, self.num_experts),
        )

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = LayerNorm(config.hidden_size)

        # Adaptive sparsity threshold
        self.sparsity_threshold = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Route input to experts
        routing_weights = F.softmax(self.router(x), dim=-1)

        # Apply adaptive sparsity
        mask = (routing_weights > self.sparsity_threshold).float()
        routing_weights = routing_weights * mask
        routing_weights = routing_weights / (
            routing_weights.sum(dim=-1, keepdim=True) + 1e-6
        )

        # Combine expert outputs
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            output += routing_weights[..., i : i + 1] * expert_output

        return self.layer_norm(residual + self.dropout(output))


class MultiScaleAttention(nn.Module):
    """Multi-scale attention with fixed dimension handling"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size

        # Single set of projections instead of multiple
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Scale mixing parameters
        self.scales = (
            [1, 2, 4]
            if not hasattr(config, "attention_scales")
            else config.attention_scales
        )
        self.scale_weights = nn.Parameter(
            torch.ones(len(self.scales)) / len(self.scales)
        )

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = LayerNorm(config.hidden_size)

    def _apply_pooling(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Apply pooling while preserving tensor dimensions"""
        if scale == 1:
            return x

        B, H, L, D = x.shape
        # Pad the sequence length dimension if needed
        pad_len = (scale - L % scale) % scale
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        # Reshape and pool
        x = x.view(B, H, -1, scale, D).mean(dim=3)
        return x

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.size()

        # Single projection for q, k, v
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        outputs = []
        scale_weights = F.softmax(self.scale_weights, dim=0)

        for i, scale in enumerate(self.scales):
            # Apply pooling to k and v while maintaining proper dimensions
            k_scaled = self._apply_pooling(k, scale)
            v_scaled = self._apply_pooling(v, scale)

            # Adjust attention mask if needed
            if mask is not None:
                scaled_mask = mask
                if scale > 1:
                    # Pool the mask as well
                    scaled_mask = mask.view(B, 1, -1, scale).any(dim=-1)
                    scaled_mask = scaled_mask.unsqueeze(1).unsqueeze(2)
            else:
                scaled_mask = None

            # Compute attention
            attn_output = F.scaled_dot_product_attention(
                q,
                k_scaled,
                v_scaled,
                attn_mask=scaled_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )

            outputs.append(attn_output * scale_weights[i])

        # Combine outputs from different scales
        combined_output = sum(outputs)
        combined_output = combined_output.transpose(1, 2).contiguous().view(B, L, D)

        return self.layer_norm(x + self.dropout(self.out_proj(combined_output)))


class EnhancedDomainAdapter(nn.Module):
    """Enhanced domain adapter with meta-learning capabilities and fixed dimensions"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.domain_dim = config.domain_dim

        # Meta-learner with corrected output dimensions
        self.meta_adapter = nn.Sequential(
            LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            # Adjust output dimension to match expected size
            nn.Linear(config.hidden_size, config.domain_dim),
        )

        # Learned temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))

        # Domain prototypes
        self.domain_prototypes = nn.Parameter(
            torch.randn(config.num_domains, config.domain_dim)
            / math.sqrt(config.domain_dim)
        )

        # Task-specific adaptation
        self.task_adapter = nn.Sequential(
            LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.adapter_dim),
            nn.ReLU(),
            nn.Linear(config.adapter_dim, config.hidden_size),
            LayerNorm(config.hidden_size),
        )

        self.adapter_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape

        # Generate domain query directly without additional matrix multiplication
        domain_query = self.meta_adapter(x)  # Shape: [B, L, domain_dim]

        # Compute domain similarities
        domain_scores = torch.matmul(
            domain_query, self.domain_prototypes.T
        )  # Shape: [B, L, num_domains]

        # Apply learned temperature scaling
        domain_probs = F.softmax(domain_scores / self.temperature, dim=-1)

        # Task-specific adaptation
        task_output = self.task_adapter(x)
        output = x + self.adapter_scale * task_output

        return output, domain_probs


class EnhancedTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.domain_adapter = EnhancedDomainAdapter(config)
        self.attention = MultiScaleAttention(config)
        self.mlp = AdaptiveSparseMLP(config)
        self.prenorm = LayerNorm(config.hidden_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.prenorm(x)

        # Enhanced domain adaptation
        adapted_x, domain_probs = self.domain_adapter(x)

        # Multi-scale attention
        attn_output = self.attention(adapted_x, mask)

        # Sparse MLP with mixture of experts
        output = self.mlp(attn_output)

        return output + residual, domain_probs


class EnhancedGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Enhanced token embedding with learned scaling
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_scale = nn.Parameter(torch.ones(1) * 0.05)
        nn.init.normal_(self.token_emb.weight, std=0.05)

        # Relative position embeddings
        self.max_positions = config.max_seq_length
        self.pos_emb = nn.Parameter(
            torch.zeros(2 * config.max_seq_length - 1, config.hidden_size)
        )
        nn.init.normal_(self.pos_emb, std=0.05)

        self.dropout = nn.Dropout(config.dropout)

        # Create enhanced transformer blocks
        self.blocks = nn.ModuleList(
            [EnhancedTransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.ln_final = LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.output.weight, std=0.05)

    def get_relative_positions(self, seq_length: int) -> torch.Tensor:
        positions = torch.arange(seq_length, device=torch.device("cuda"))
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions += self.max_positions - 1  # Shift to positive indices
        return relative_positions

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, L = x.size()

        # Apply learned embedding scale
        x = self.token_emb(x) * self.emb_scale

        # Add relative position embeddings
        relative_positions = self.get_relative_positions(L)
        pos_emb = self.pos_emb[relative_positions]
        x = x + pos_emb.mean(dim=1)

        x = self.dropout(x)

        # Process through enhanced transformer blocks
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
        num_domains: int = 384,
        adapter_dim: int = 384,
        init_std: float = 0.05,
        # New parameters for enhanced architecture
        num_experts: int = 4,
        expert_capacity: float = 1.0,  # Capacity factor for routing
        attention_scales: List[int] = [1, 2, 4],  # Multi-scale attention windows
        meta_learning_rate: float = 0.01,  # Meta-learning adaptation rate
        temperature_init: float = 1.0,  # Initial temperature for domain routing
        sparsity_target: float = 0.3,  # Target sparsity for adaptive MLPs
        relative_pos_buckets: int = 32,  # Number of relative position buckets
        use_adaptive_sparsity: bool = True,  # Whether to use adaptive sparsity
        use_multi_scale_attention: bool = True,  # Whether to use multi-scale attention
        meta_adapter_hidden_size: int = 1024,  # Hidden size for meta adapter
        router_z_loss_coef: float = 0.001,  # Router z-loss coefficient
        router_aux_loss_coef: float = 0.001,  # Router auxiliary loss coefficient
    ):
        # Validate configuration
        assert (
            hidden_size % num_heads == 0
        ), "Hidden size must be divisible by number of heads"
        assert num_experts > 0, "Number of experts must be positive"
        assert 0.0 < expert_capacity <= 2.0, "Expert capacity must be between 0 and 2"
        assert all(s > 0 for s in attention_scales), "Attention scales must be positive"
        assert 0.0 <= sparsity_target < 1.0, "Sparsity target must be between 0 and 1"

        # Base parameters
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

        # Expert parameters
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef

        # Attention parameters
        self.attention_scales = attention_scales
        self.head_dim = hidden_size // num_heads
        self.use_multi_scale_attention = use_multi_scale_attention

        # Domain adaptation parameters
        self.meta_learning_rate = meta_learning_rate
        self.temperature_init = temperature_init
        self.meta_adapter_hidden_size = meta_adapter_hidden_size

        # Sparsity parameters
        self.sparsity_target = sparsity_target
        self.use_adaptive_sparsity = use_adaptive_sparsity

        # Position embedding parameters
        self.relative_pos_buckets = relative_pos_buckets

        # Derived parameters
        self.total_num_heads = (
            num_heads * len(attention_scales)
            if use_multi_scale_attention
            else num_heads
        )
        self.max_relative_position = 2 * max_seq_length - 1

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_heads

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GPTConfig":
        """Create config from dictionary"""
        return cls(**config_dict)

    def __repr__(self) -> str:
        """Pretty print config"""
        attrs = [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        return f"GPTConfig({', '.join(attrs)})"


class FineWebDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512):
        # Load the FineWeb dataset
        self.dataset = load_dataset("deatos/fineweb-edu-10b-combined", split=split)
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

        # Decode truncated tokens back to text
        truncated_text = self.tokenizer.decode(
            truncated_tokens, skip_special_tokens=True
        )

        # Tokenize the truncated text for the final input
        encodings = self.tokenizer(
            truncated_text,
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
            "attention_mask": attention_mask,
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

        # Tokenize the response and truncate it at a random index
        response_tokens = self.tokenizer(
            item["response"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )["input_ids"].squeeze(0)

        # Random truncation
        truncation_idx = random.randint(1, len(response_tokens))
        truncated_response = response_tokens[:truncation_idx]

        # Decode truncated response back to text for concatenation
        truncated_response_text = self.tokenizer.decode(
            truncated_response, skip_special_tokens=True
        )

        # Combine instruction, context, and truncated response
        text = f"Instruction: {item['instruction']}\n"
        if item["context"]:
            text += f"Context: {item['context']}\n"
        text += f"Response: {truncated_response_text}"

        # Tokenize the full sequence
        encodings = self.tokenizer(
            text,
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
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_model(
    model,
    tokenizer,
    train_dataloader,
    val_dataloader,
    num_epochs=3,
    learning_rate=3e-4,
    eta_min=1e-6,
    max_grad_norm=1.0,
    warmup_steps=1000,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Training on device: {device}")
    model = model.to(device)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_dataloader), eta_min=eta_min
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training statistics
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    grads = None

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
            logits, domain = model(input_ids)

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)

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

                logits, domain = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
                )

                total_val_loss += loss.item()
                val_pbar.set_postfix({"loss": loss.item()})

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

    return train_losses, val_losses


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


def split_ds(ds, train_perc=0.9):
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
        train_dataset, batch_size=24, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=24, shuffle=False, num_workers=4
    )
    return train_dataloader, val_dataloader


def main():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create config and model
    config = GPTConfig(
        max_seq_length=512,
        hidden_size=384,
        num_layers=26,
        num_heads=8,
        dropout=0.1,
        domain_dim=256,
        num_domains=2,
        adapter_dim=256,
        num_experts=2,
        meta_adapter_hidden_size=256,
    )
    model = EnhancedGPT(config)
    model = torch.compile(model, backend="cudagraphs")

    # model.load_state_dict(torch.load("gpt2_crosschop.pth"))

    # Create full dataset
    full_dataset = FineWebDataset("train[:10%]", tokenizer)

    # Train model
    train_dataloader, val_dataloader = split_ds(full_dataset)

    train_losses, val_losses = train_model(
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        num_epochs=4,
        learning_rate=5e-4,
        eta_min=5e-6,
    )

    full_dataset = DollyDataset("train", tokenizer=tokenizer)

    # Freeze first 4 layers
    for layer in model.blocks[:4]:
        for param in layer.parameters():
            param.requires_grad = False

    train_dataloader, val_dataloader = split_ds(full_dataset)
    ft_train_losses, ft_val_losses = train_model(
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        num_epochs=4,
        learning_rate=1e-5,
    )
    torch.save(model.state_dict(), "gpt2_crosschop.pth")
    print("Saved model checkpoint")
    # Test generation
    test_prompt = (
        "Instruction: Explain what machine learning is to a 5-year old.\nResponse:"
    )
    generated_text = generate_text(model, tokenizer, test_prompt)
    print("\nTest Generation:")
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")

    # Save Model


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model, GPT2MLP
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from optimum.bettertransformer import BetterTransformer


#  TBD items
#  Port domain adapter and attention mechanisms into custom module package
#  FT dataset
#
#
class MemoryWeightedAttention(nn.Module):
    """Adds retrieval-weighted attention with compressed valence representations."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        context_size=1024,
        epsilon=0.066,
        ffn_dim=512,
        valence_dim=288,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.context_size = context_size
        self.epsilon = epsilon
        self.valence_dim = valence_dim

        # Memory tracking for token occurrences
        self.register_buffer("token_occurrence", torch.zeros(context_size))

        # Compressed valence embedding
        self.valence_embedding = nn.Embedding(context_size, valence_dim)
        nn.init.xavier_uniform_(self.valence_embedding.weight)

        # Projection from compressed valence to attention heads
        self.valence_projection = nn.Sequential(
            nn.Linear(valence_dim, valence_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(valence_dim * 2, num_heads),
            nn.Tanh(),  # Bounded output
        )

        # Feed-Forward Network for processing combined weights
        self.ffn = nn.Sequential(
            nn.Linear(num_heads + 1, ffn_dim),  # +1 for occurrence weight
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, ffn_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim // 2, 1),
            nn.Tanh(),  # Output in [-1, 1] range for stability
        )

        # Layer Normalization for feature normalization
        self.layer_norm_valence = nn.LayerNorm(valence_dim)
        self.layer_norm_combined = nn.LayerNorm(num_heads + 1)

    def forward(self, attention_scores, input_ids):
        """Modifies attention scores based on token frequency and compressed valence."""
        batch_size, num_heads, seq_len, _ = attention_scores.shape

        # Make sure input_ids is contiguous and properly reshaped
        input_ids = input_ids.contiguous()
        flattened_ids = input_ids.reshape(-1)

        # Get token occurrences in the batch
        token_counts = torch.bincount(
            flattened_ids, minlength=self.context_size
        ).float()
        token_counts = token_counts.clamp(min=1)

        # Update moving memory with decay
        decay_factor = 0.87
        self.token_occurrence.mul_(decay_factor).add_(token_counts)

        # Compute occurrence weights
        occurrence_weights = torch.log1p(self.token_occurrence[input_ids])

        # Get compressed valence embeddings and project to attention heads
        valence_compressed = self.valence_embedding(
            input_ids
        )  # [batch_size, seq_len, valence_dim]
        valence_compressed = self.layer_norm_valence(valence_compressed)

        # Process each position in the sequence for valence projection
        batch_seq_size = batch_size * seq_len
        flattened_valence = valence_compressed.view(batch_seq_size, -1)

        # Project to attention head space
        valence_scores = self.valence_projection(flattened_valence)
        valence_scores = valence_scores.view(batch_size, seq_len, self.num_heads)

        # Prepare inputs for FFN
        occurrence_weights_expanded = occurrence_weights.unsqueeze(
            -1
        )  # [batch_size, seq_len, 1]

        # Concatenate occurrence weights with projected valence scores
        combined_features = torch.cat(
            [occurrence_weights_expanded, valence_scores], dim=-1
        )  # [batch_size, seq_len, num_heads + 1]

        # Apply layer normalization
        normalized_features = self.layer_norm_combined(combined_features)

        # Process through FFN
        flattened_features = normalized_features.view(batch_seq_size, -1)
        attention_modifiers = self.ffn(flattened_features)  # [batch_size * seq_len, 1]

        # Reshape back to match attention scores
        attention_modifiers = attention_modifiers.view(batch_size, seq_len, 1, 1)
        attention_modifiers = attention_modifiers.expand(-1, -1, num_heads, 1)
        attention_modifiers = attention_modifiers.permute(
            0, 2, 1, 3
        )  # [batch_size, num_heads, seq_len, 1]

        # Clip weights to be within [1-ε, 1+ε]
        retrieval_weights = 1.0 + self.epsilon * attention_modifiers

        # Apply to attention scores
        modified_attention_scores = attention_scores * retrieval_weights

        return modified_attention_scores


class ModifiedGPT2Attention(GPT2Attention):
    """GPT-2 Attention modified to include enhanced Memory-Weighted Attention"""

    def __init__(self, config):
        super().__init__(config)
        self.mwa_module = MemoryWeightedAttention(
            config.hidden_size,
            config.num_attention_heads,
            epsilon=0.1,
            ffn_dim=256,
            valence_dim=32,  # Compressed valence dimension
        )

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / torch.sqrt(
            torch.tensor(value.size(-1), dtype=torch.float)
        )

        # Get the input shape
        batch_size = query.size(0)
        seq_len = query.size(2)

        # Create input_ids tensor with proper shape and make it contiguous
        input_ids = (
            torch.arange(seq_len, device=query.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )

        # Apply enhanced memory-weighted attention
        attn_weights = self.mwa_module(attn_weights, input_ids)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class ValenceModule(nn.Module):
    def __init__(self, config, num_domains):
        super().__init__()
        self.hidden_size = config.n_embd

        # Learnable domain embeddings
        self.domain_embeddings = nn.Embedding(num_domains, self.hidden_size // 2)
        self.domain_projection = nn.Sequential(
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Valence matrix
        self.valence_matrix = nn.Parameter(
            torch.randn(self.hidden_size, self.hidden_size) * 0.01
        )

        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x, domain_ids):
        # Retrieve domain-specific embeddings
        domain_embeds = self.domain_embeddings(domain_ids)  # (batch_size, hidden_size)
        domain_proj = self.domain_projection(domain_embeds)
        # Project valence matrix using domain embeddings
        projected_valence = torch.matmul(
            domain_proj, self.valence_matrix
        )  # (batch_size, hidden_size)

        # Apply valence transformation
        valence_scores = torch.matmul(x, projected_valence.unsqueeze(-1)).squeeze(-1)
        x = x + F.tanh(valence_scores)  # Adjust x with valence-based modifications
        x = self.norm(x)  # Normalize

        return x


class ModifiedGPTMLP(GPT2MLP):
    def __init__(self, config):
        super().__init__(config)
        self.valence_module = ValenceModule(
            config, num_domains=8
        )  # Insert ValenceModule

    def forward(self, x):
        x = self.valence_module(x)  # Apply valence tracking
        return super().forward(x)  # Continue with FFN


class ModifiedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.h):
            layer.attn = ModifiedGPT2Attention(config)
            layer.mlp = ModifiedGPTMLP(config)


class FineWebDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512):
        self.dataset = load_dataset("deatos/fineweb-edu-10b-combined", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.bos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get text and create input-target pairs for next token prediction
        text = self.dataset[idx]["text"]

        # Tokenize with proper truncation and padding
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Create labels by shifting input_ids right
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.tokenizer.eos_token_id

        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def freeze_except_modules(model, module_names=["custom_module"]):
    """"""
    if isinstance(module_names, str):
        module_names = [module_names]  # Ensure it's always a list

    for name, param in model.named_parameters():
        if any(
            module in name for module in module_names
        ):  # Check if any module matches
            param.requires_grad = True  # Keep module trainable
        else:
            param.requires_grad = False  # Freeze everything else


model_name = "gpt2"  # Use "gpt2-medium" or other variants if needed
model = GPT2LMHeadModel.from_pretrained(model_name)

# Modify GPT-2 by replacing attention layers
for i, layer in enumerate(model.transformer.h):
    layer.attn = ModifiedGPT2Attention(model.config)

# Freeze all weights except the new module
freeze_except_modules(model, module_names=["mwa_module", "valence_module"])


tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
train_data = FineWebDataset(split="train[:12%]", tokenizer=tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    num_train_epochs=4,  # Short run
    per_device_train_batch_size=24,
    save_steps=250,
    save_total_limit=2,
    learning_rate=2e-5,  # Adjust learning rate for the module
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator,
)

trainer.train()

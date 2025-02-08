import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from datasets import load_dataset, concatenate_datasets
from grokfast import gradfilter_ema
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple, List, Dict, Any
import random
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb


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
    """Multi-scale attention with proper mask handling"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

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
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def _pool_mask(self, mask: torch.Tensor, scale: int) -> torch.Tensor:
        """Pool attention mask to match scaled key/value sequence length"""
        if scale == 1:
            return mask

        B, H, L, S = mask.shape
        pad_len = (scale - S % scale) % scale

        if pad_len > 0:
            # Pad the mask
            mask = F.pad(mask, (0, pad_len), value=float("-inf"))

        # Reshape and pool
        new_S = (S + pad_len) // scale
        mask = mask.view(B, H, L, new_S, scale)
        # Use max pooling to preserve valid attention positions
        mask = mask.amax(dim=-1)
        return mask

    def _apply_pooling(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Apply pooling while preserving tensor dimensions"""
        if scale == 1:
            return x

        B, H, L, D = x.shape
        pad_len = (scale - L % scale) % scale

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        pooled_length = (L + pad_len) // scale
        x = x.view(B, H, pooled_length, scale, D).mean(dim=3)
        return x

    def _prepare_mask(self, mask: torch.Tensor, tgt_len: int) -> torch.Tensor:
        """Prepare attention mask for scaled dot product attention"""
        # Convert from [batch_size, seq_len] to [batch_size, num_heads, tgt_len, src_len]
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, self.num_heads, tgt_len, -1)

        # Convert boolean/binary mask to attention mask
        return mask.to(dtype=torch.float32).masked_fill(~mask.bool(), float("-inf"))

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.size()

        # Project queries, keys, and values
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        outputs = []
        scale_weights = F.softmax(self.scale_weights, dim=0)

        # Prepare base attention mask if provided
        if attention_mask is not None:
            attention_mask = self._prepare_mask(attention_mask, L)

        for i, scale in enumerate(self.scales):
            # Apply pooling to keys and values
            k_scaled = self._apply_pooling(k, scale)
            v_scaled = self._apply_pooling(v, scale)

            # Handle attention mask for this scale
            if attention_mask is not None:
                # Pool the mask to match the scaled key sequence length
                scaled_mask = self._pool_mask(attention_mask, scale)
            else:
                scaled_mask = None

            # Compute attention
            attn_output = F.scaled_dot_product_attention(
                q,
                k_scaled,
                v_scaled,
                attn_mask=scaled_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )

            outputs.append(attn_output * scale_weights[i])

        combined_output = sum(outputs)
        combined_output = combined_output.transpose(1, 2).contiguous().view(B, L, D)

        return self.layer_norm(x + self.dropout(self.out_proj(combined_output)))


# First, create a shared embedding buffer manager
class SharedEmbeddingBuffer:
    def __init__(self, max_history: int, domain_dim: int, device: torch.device):
        self.max_history = max_history
        self.domain_dim = domain_dim

        # Initialize shared buffers
        self.embedding_history = torch.zeros(max_history, domain_dim, device=device)
        self.history_mask = torch.zeros(max_history, dtype=torch.bool, device=device)
        self.history_pointer = torch.tensor(0, dtype=torch.long, device=device)

    def update(self, new_embeddings: torch.Tensor):
        """Update shared embedding history"""
        batch_size = new_embeddings.size(0)
        space_left = self.max_history - self.history_pointer
        embeddings_to_add = min(batch_size, space_left)

        # Add embeddings to the available space
        if embeddings_to_add > 0:
            self.embedding_history[
                self.history_pointer : self.history_pointer + embeddings_to_add
            ] = new_embeddings[:embeddings_to_add]
            self.history_mask[
                self.history_pointer : self.history_pointer + embeddings_to_add
            ] = True
            self.history_pointer += embeddings_to_add

        # Handle wrap-around for the remaining embeddings
        remaining = batch_size - embeddings_to_add
        if remaining > 0:
            wrap_indices = remaining % self.max_history  # Fit within buffer size
            if remaining > self.max_history:
                # Only keep the most recent `self.max_history` embeddings
                new_embeddings = new_embeddings[-self.max_history :]
                remaining = self.max_history
                wrap_indices = 0  # All embeddings will be replaced

            self.embedding_history[:remaining] = new_embeddings[embeddings_to_add:]
            self.history_mask[:remaining] = True
            self.history_pointer = wrap_indices


class LayerSpecificDomainAdapter(nn.Module):
    """Domain adapter with proper shape handling"""

    def __init__(self, config, shared_buffer: SharedEmbeddingBuffer, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.domain_dim = config.domain_dim
        self.num_heads = 4
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.dropout = config.dropout

        self.shared_buffer = shared_buffer

        # Ensure meta adapter output matches domain dimension
        self.meta_adapter = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.meta_adapter_hidden_size),
            nn.ReLU(),
            nn.Linear(config.meta_adapter_hidden_size, config.domain_dim),
        )

        # Project from domain dimension to hidden size
        self.q_proj = nn.Linear(config.domain_dim, config.hidden_size)
        self.k_proj = nn.Linear(config.domain_dim, config.hidden_size)
        self.v_proj = nn.Linear(config.domain_dim, config.hidden_size)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.temperature = nn.Parameter(
            torch.tensor([config.temperature_init], device="cuda:0")
        )

        # Domain prototypes now match domain dimension
        self.domain_prototypes = nn.Parameter(
            torch.randn(config.num_domains, config.domain_dim)
            / math.sqrt(config.domain_dim)
        )

        self.task_adapter = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.adapter_dim),
            nn.ReLU(),
            nn.Linear(config.adapter_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

        self.adapter_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.kv_cache = None

    def _prepare_domain_mask(
        self, attention_mask: torch.Tensor, past_length: int = 0
    ) -> torch.Tensor:
        """Prepare attention mask for domain attention"""
        B = attention_mask.size(0)
        L = attention_mask.size(-1)

        # Create mask for domain states
        if past_length > 0:
            domain_mask = torch.ones((B, past_length), device=attention_mask.device)
            attention_mask = torch.cat([domain_mask, attention_mask], dim=1)

        # Expand mask for multi-head attention
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(B, self.num_heads, L, -1)

        return attention_mask.to(dtype=torch.float32) * -1e9

    def get_relevant_embeddings(
        self,
        query_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_domain_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, L, D = query_states.shape
        past_length = 0 if past_domain_state is None else past_domain_state.size(1)

        # Project query, key, and value while maintaining proper shapes
        q = (
            self.q_proj(query_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(query_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(query_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if past_domain_state is not None:
            past_k = (
                self.k_proj(past_domain_state)
                .view(B, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            past_v = (
                self.v_proj(past_domain_state)
                .view(B, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_domain_mask(attention_mask, past_length)

        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Process output
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        )
        attn_output = self.out_proj(attn_output)

        if self.config.use_cache:
            self.kv_cache = (k, v)
            return attn_output, (k, v)
        return attn_output, None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_domain_state: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        if past_key_values is not None:
            self.kv_cache = past_key_values

        # Get domain query representations
        domain_query = self.meta_adapter(x)  # [B, L, domain_dim]
        enhanced_query, past_kv = self.get_relevant_embeddings(
            domain_query,
            attention_mask,
            past_domain_state,
        )

        # Compute domain scores with proper reshaping
        # First average over sequence length to get per-batch embeddings
        batch_domain_query = domain_query.mean(dim=1)  # [B, domain_dim]

        # Now compute domain scores with correct shapes
        domain_scores = torch.matmul(
            batch_domain_query, self.domain_prototypes.T
        )  # [B, num_domains]
        domain_probs = F.softmax(domain_scores / self.temperature, dim=-1)

        # Update shared buffer for first layer
        if self.layer_idx == 0:
            with torch.no_grad():
                # Use the averaged domain query for the buffer
                self.shared_buffer.update(batch_domain_query)

        task_output = self.task_adapter(x)
        output = x + self.adapter_scale * task_output

        if self.config.use_cache:
            return output, domain_probs, past_kv
        return output, domain_probs


class EnhancedTransformerBlock(nn.Module):
    def __init__(self, config, shared_buffer: SharedEmbeddingBuffer, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.domain_adapter = LayerSpecificDomainAdapter(
            config, shared_buffer, layer_idx
        )
        self.attention = MultiScaleAttention(config)
        self.mlp = AdaptiveSparseMLP(config)
        self.prenorm = LayerNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_domain_state: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x = self.prenorm(x)

        # Handle domain adapter with caching
        if self.config.use_cache:
            adapted_x, domain_probs, domain_kv = self.domain_adapter(
                x,
                attention_mask=attention_mask,
                past_key_values=past_key_values[0] if past_key_values else None,
            )
            # Handle attention with caching
            attn_output = self.attention(
                adapted_x,
                attention_mask,
            )
            output = self.mlp(attn_output)
            if isinstance(attn_output, tuple) and len(attn_output) > 1:
                return output + residual, domain_probs, (domain_kv, attn_output[1])
            else:
                return (output + residual, domain_probs, (domain_kv, None))

        # Non-caching forward pass
        adapted_x, domain_probs = self.domain_adapter(
            x, attention_mask=attention_mask, past_domain_state=past_domain_state
        )
        attn_output = self.attention(adapted_x, attention_mask)
        output = self.mlp(attn_output)
        return (
            output + residual,
            domain_probs,
        )


class EnhancedGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create shared embedding buffer
        self.shared_buffer = SharedEmbeddingBuffer(
            max_history=getattr(config, "max_domain_history", 1000),
            domain_dim=config.domain_dim,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Token embedding and scaling
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_scale = nn.Parameter(torch.ones(1) * 0.05)
        nn.init.normal_(self.token_emb.weight, std=0.05)

        # Position embeddings
        self.max_positions = config.max_seq_length
        self.pos_emb = nn.Parameter(
            torch.zeros(2 * config.max_seq_length - 1, config.hidden_size)
        )
        nn.init.normal_(self.pos_emb, std=0.05)

        self.dropout = nn.Dropout(config.dropout)

        # Create transformer blocks with shared buffer
        self.blocks = nn.ModuleList(
            [
                EnhancedTransformerBlock(config, self.shared_buffer, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )

        self.ln_final = LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.output.weight, std=0.05)

    def get_relative_positions(self, seq_length: int) -> torch.Tensor:
        positions = torch.arange(seq_length, device=self.pos_emb.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions += self.max_positions - 1
        return relative_positions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_domain_states: Optional[List[torch.Tensor]] = None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[
            torch.Tensor, List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]
        ],
    ]:

        B, L = input_ids.shape

        # Get embeddings and positions
        x = self.token_emb(input_ids) * self.emb_scale

        # Add relative position embeddings
        relative_positions = self.get_relative_positions(L)
        pos_emb = self.pos_emb[relative_positions]
        pos_emb_mean = pos_emb.mean(dim=1)
        x = x + pos_emb_mean

        x = self.dropout(x)

        # Initialize lists for collecting states
        domain_states = [] if self.config.return_domain_states else None
        past_key_values = [] if self.config.use_cache else None

        # Process through transformer blocks
        for idx, block in enumerate(self.blocks):
            # Get past domain state for this layer if available
            past_domain_state = None
            if past_domain_states is not None and idx < len(past_domain_states):
                past_domain_state = past_domain_states[idx]

            if self.config.use_cache:
                x, domain_probs, (past_k, past_v) = block(
                    x,
                    attention_mask=attention_mask,
                    past_domain_state=past_domain_state,
                    past_key_values=past_key_values,
                )
                past_key_values.append((past_k, past_v))
            else:
                x, domain_probs = block(
                    x, mask=attention_mask, past_domain_state=past_domain_state
                )

            if self.config.return_domain_states:
                domain_states.append(domain_probs)

        x = self.ln_final(x)
        logits = self.output(x)

        # Return based on flags
        if self.config.use_cache:
            return logits, domain_states, past_key_values
        elif self.config.return_domain_states:
            return logits, domain_states
        else:
            return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        min_new_tokens: int = 1,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text based on input prompt with proper domain adaptation and KV caching.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            max_new_tokens: Maximum number of tokens to generate
            min_new_tokens: Minimum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
        Returns:
            torch.Tensor: Generated token IDs
        """
        if pad_token_id is None:
            pad_token_id = 0  # Default padding token
        if eos_token_id is None:
            eos_token_id = 2  # Default EOS token

        # Initialize generation variables
        batch_size = input_ids.shape[0]
        device = input_ids.device
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        generated = input_ids

        # Initialize domain states and key-value cache
        past_domain_states = None
        past_key_values = None

        # Track banned ngrams for no_repeat_ngram_size
        banned_ngrams = [{} for _ in range(batch_size)]

        # Generate tokens up to max_length
        cur_len = input_ids.shape[1]
        min_length = cur_len + min_new_tokens
        max_length = cur_len + max_new_tokens

        while cur_len < max_length:
            # Forward pass with domain state and KV cache tracking
            outputs = self.forward(
                generated,
                attention_mask=attention_mask,
                past_domain_states=past_domain_states,
            )
            logits, domain_states, current_key_values = outputs

            # Update states for next iteration
            past_domain_states = domain_states
            past_key_values = current_key_values

            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(generated[i].tolist()):
                        next_token_logits[i, previous_token] /= repetition_penalty

            # Handle no-repeat ngrams
            if no_repeat_ngram_size > 0:
                for batch_idx in range(batch_size):
                    # Get the current sequence
                    current_seq = generated[batch_idx].tolist()
                    # Create ngrams from the last no_repeat_ngram_size-1 tokens
                    if len(current_seq) >= no_repeat_ngram_size:
                        ngram_prefix = tuple(current_seq[-(no_repeat_ngram_size - 1) :])
                        # If this prefix exists in banned_ngrams, ban its next tokens
                        if ngram_prefix in banned_ngrams[batch_idx]:
                            banned_tokens = banned_ngrams[batch_idx][ngram_prefix]
                            next_token_logits[batch_idx, banned_tokens] = float("-inf")

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][
                        sorted_indices_to_remove[batch_idx]
                    ]
                    next_token_logits[batch_idx, indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(probs, dim=-1, keepdim=True)

            # Handle no-repeat ngrams bookkeeping
            if no_repeat_ngram_size > 0:
                for batch_idx in range(batch_size):
                    current_seq = generated[batch_idx].tolist() + [
                        next_tokens[batch_idx, 0].item()
                    ]
                    # Only process if sequence is long enough
                    if len(current_seq) >= no_repeat_ngram_size:
                        # Create and store ngram in banned_ngrams
                        ngram = tuple(current_seq[-(no_repeat_ngram_size - 1) :])
                        next_token = current_seq[-1]
                        if ngram not in banned_ngrams[batch_idx]:
                            banned_ngrams[batch_idx][ngram] = []
                        banned_ngrams[batch_idx][ngram].append(next_token)

            # Mark sequences as finished if they hit EOS
            unfinished_sequences = unfinished_sequences.mul(
                (next_tokens != eos_token_id).long()
            )

            # Stop if all sequences are finished and we're past min_length
            if unfinished_sequences.max() == 0 and cur_len >= min_length:
                break

            # Update sequences and attention mask
            generated = torch.cat([generated, next_tokens], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
                )

            cur_len = generated.shape[1]

        return generated


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
        max_domain_history=64,
        expert_capacity: float = 1.0,  # Capacity factor for routing
        attention_scales: List[int] = [1, 2, 4],  # Multi-scale attention windows
        meta_learning_rate: float = 0.01,  # Meta-learning adaptation rate
        temperature_init: float = 0.7,  # Initial temperature for domain routing
        sparsity_target: float = 0.3,  # Target sparsity for adaptive MLPs
        relative_pos_buckets: int = 32,  # Number of relative position buckets
        use_adaptive_sparsity: bool = True,  # Whether to use adaptive sparsity
        use_multi_scale_attention: bool = True,  # Whether to use multi-scale attention
        use_cache: bool = True,
        return_domain_states: bool = True,
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
        self.max_domain_history = max_domain_history
        self.use_cache = use_cache
        self.return_domain_states = return_domain_states

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
        self.dataset = load_dataset("deatos/fineweb-edu-10b-combined", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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


class InstructDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512):
        # Combine instruction datasets
        ds = (
            load_dataset("iamtarun/python_code_instructions_18k_alpaca", split=split)
            .rename_column("input", "context")
            .rename_column("output", "response")
        )
        ds2 = load_dataset(
            "ewof/code-alpaca-instruct-unfiltered", split=split
        ).rename_column("output", "response")

        ds3 = (
            load_dataset(
                "HuggingFaceH4/helpful_instructions",
                split="all",
                trust_remote_code=True,
            )
            .rename_column("prompt", "instruction")
            .rename_column("completion", "response")
        )
        self.dataset = concatenate_datasets(
            [ds, ds2, ds3, load_dataset("databricks/databricks-dolly-15k", split=split)]
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        special_tokens = {"additional_special_tokens": ["<instruct>", "<response>"]}

        # Add the special tokens to the tokenizer
        self.tokenizer.add_special_tokens(special_tokens)

        # Special tokens for formatting
        self.sep_token = "\n"
        self.end_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Format instruction input
        instruction_text = f"<instruct> {item['instruction']}{self.sep_token}"
        if item["context"]:
            instruction_text += f"Context: {item['context']}{self.sep_token}"
        instruction_text += f"<response> "

        # Format full sequence with response
        full_text = instruction_text + item["response"] + self.end_token

        # Tokenize full sequence
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Find response start token position
        response_start = len(self.tokenizer(instruction_text)["input_ids"])

        # Create labels with -100 for instruction tokens
        labels = input_ids.clone()
        labels[-1] = self.tokenizer.eos_token_id

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_model(
    model: EnhancedGPT,
    tokenizer: AutoTokenizer,
    full_dataset,
    num_epochs=4,
    learning_rate=3e-4,
    eta_min=1e-6,
    max_grad_norm=1.0,
    warmup_steps=1000,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    num_samples = int(len(full_dataset) * 0.05)
    all_indices = list(range(len(full_dataset)))
    random_indices = random.sample(all_indices, num_samples)  # Randomly sample indices
    subset = Subset(full_dataset, random_indices)
    train_dataloader, val_dataloader = split_ds(subset)

    print(f"Training on device: {device}")
    model = model.to(device)

    # Initialize optimizer and scheduler
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=500,
        T_mult=2,
        eta_min=eta_min,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training statistics
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        if epoch > 0:
            all_indices = list(range(len(full_dataset)))
            random_indices = random.sample(
                all_indices, num_samples
            )  # Randomly sample indices
            subset = Subset(full_dataset, random_indices)
            train_dataloader, val_dataloader = split_ds(subset)

        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Track domain states and key-value cache
        batch_domain_states = None
        batch_past_kv = None

        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].bool().to(device)
            labels = batch["labels"].to(device)

            # Forward pass with domain and cache tracking
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                past_domain_states=batch_domain_states,
            )

            logits, domain_states, past_kv = outputs

            # Update domain states and cache for next batch
            batch_domain_states = domain_states
            if model.config.use_cache:
                batch_past_kv = past_kv

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Add domain consistency loss if applicable
            if batch_idx > 0 and domain_states is not None:
                domain_consistency_loss = sum(
                    F.kl_div(curr.log(), prev, reduction="batchmean")
                    for curr, prev in zip(domain_states, batch_domain_states)
                )
                loss = loss + 0.1 * domain_consistency_loss

            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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

        # Reset states for validation
        val_domain_states = None
        val_past_kv = None

        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    past_domain_states=val_domain_states,
                )

                logits, domain_states, past_kv = outputs

                # Update states
                val_domain_states = domain_states
                if model.config.use_cache:
                    val_past_kv = past_kv

                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
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
            outputs, doms, (k) = model(input_ids)
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
        train_dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=True
    )
    return train_dataloader, val_dataloader


def test_model_generation(model, tokenizer, prompt, device="cuda"):
    # Ensure model is in eval mode
    model.eval()

    # Prepare the prompt
    tokenizer_out = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Move tensors to device
    input_ids = tokenizer_out["input_ids"].to(device)
    attention_mask = tokenizer_out["attention_mask"].to(device)

    # Set generation parameters
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": 150,  # Instead of max_length
        "min_new_tokens": 30,
        "temperature": 0.7,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        # Generate text
        output_ids = model.generate(**gen_kwargs)

        # Decode only the new tokens
        input_length = input_ids.size(1)
        generated_text = tokenizer.decode(
            output_ids[0][input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Get full text for display
        full_text = tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    return generated_text, full_text


def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"

    # Create config and model
    config = GPTConfig(
        vocab_size=50304,
        max_seq_length=384,
        hidden_size=384,
        num_layers=32,
        num_heads=8,
        dropout=0.15,
        domain_dim=192,
        num_domains=2,
        adapter_dim=192,
        num_experts=2,
        meta_adapter_hidden_size=384,
        max_domain_history=16,
        use_cache=True,
        return_domain_states=True,
        relative_pos_buckets=32,
    )
    model = EnhancedGPT(config)
    model = torch.compile(model, backend="cudagraphs").to("cuda")

    # model.load_state_dict(torch.load("best_model.pth")["model_state_dict"])

    # Create full dataset

    full_dataset = FineWebDataset(
        "train[:20%]", tokenizer, max_length=config.max_seq_length
    )

    # Train model
    train_losses, val_losses = train_model(
        model,
        tokenizer,
        full_dataset,
        num_epochs=12,
        learning_rate=5e-4,
        eta_min=1e-5,
    )
    full_dataset = InstructDataset(
        "train", tokenizer=tokenizer, max_length=config.max_seq_length
    )
    """
            n = 8
    k = 4
    # Freeze first n layers
    for i, layer in enumerate(model.blocks[: n + k]):
        if i in range(n):
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.domain_adapter.parameters():
                param.requires_grad = False

    """
    ft_train_losses, ft_val_losses = train_model(
        model, tokenizer, full_dataset, num_epochs=64, learning_rate=1e-5, eta_min=1e-6
    )
    # torch.save(model.state_dict(), "gpt2_crosschop.pth")
    print("Saved model checkpoint")
    # Test generation
    test_prompt = (
        "Instruction: Explain what machine learning is to a 5-year old.\nResponse:"
    )
    test_prompt = (
        "Instruction: Explain what machine learning is to a 5-year old.\nResponse:"
    )

    print("\nTest Generation:")
    print(f"Prompt: {test_prompt}")

    generated, full_text = test_model_generation(model, tokenizer, test_prompt)

    print(f"Generated Response: {generated}")
    print(f"\nFull Text:\n{full_text}")


# Save Model


if __name__ == "__main__":
    main()

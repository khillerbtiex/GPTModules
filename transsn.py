import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt  # For wavelet transforms
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
import copy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # Shape: [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # Shape: [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)  # Shape: [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [seq_length, batch_size, d_model]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class AdaptiveResonateAndFireNeuron(nn.Module):
    def __init__(
        self, input_size, output_size, threshold=1.0, decay=0.9, inhibition_strength=0.1
    ):
        super(AdaptiveResonateAndFireNeuron, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.decay = decay
        self.inhibition_strength = inhibition_strength

        # Parameters for adaptive threshold
        self.base_threshold = threshold
        self.threshold_decay = 0.9  # Decay rate for the threshold
        self.threshold_increase = 0.1  # Amount to increase threshold when neuron fires

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_size]
        batch_size, seq_length, _ = x.shape
        device = x.device

        mem_potential = torch.zeros(batch_size, self.fc.out_features, device=device)
        thresholds = torch.full(
            (batch_size, self.fc.out_features), self.base_threshold, device=device
        )

        outputs = []
        for t in range(seq_length):
            input_t = x[:, t, :]  # Shape: [batch_size, input_size]
            current = self.fc(input_t)  # Shape: [batch_size, output_size]

            # Lateral inhibition
            inhibition = self.inhibition_strength * mem_potential.sum(
                dim=1, keepdim=True
            )
            current = current - inhibition

            # Update membrane potential
            mem_potential = self.decay * mem_potential + current

            # Generate spikes
            spikes = (mem_potential >= thresholds).float()

            # Reset membrane potential where spikes occurred (more sophisticated reset)
            mem_potential = mem_potential * (1 - spikes) + spikes * (
                mem_potential - thresholds
            )

            # Adaptive thresholds: increase thresholds where spikes occurred
            thresholds = (
                self.threshold_decay * thresholds + self.threshold_increase * spikes
            )

            outputs.append(spikes.unsqueeze(1))  # Shape: [batch_size, 1, output_size]

        outputs = torch.cat(
            outputs, dim=1
        )  # Shape: [batch_size, seq_length, output_size]
        return outputs


# Shared Wavelet Decomposition with Wavelet Compression
def shared_wavelet_decompose(x, level=3, wavelet="db1", compression_threshold=0.1):
    """
    Perform shared wavelet decomposition across embedding dimensions, apply compression,
    and return factorized tensors with upsampled sequence lengths to match original.

    Args:
        x: Tensor of shape [batch_size, seq_length, embedding_dim]
        level: Decomposition level
        wavelet: Wavelet type
        compression_threshold: Threshold for wavelet compression

    Returns:
        factorized_coeffs: List of factorized tensors for each level, each of shape [batch_size, embedding_dim, seq_length]
    """
    batch_size, seq_length, embedding_dim = x.shape
    device = x.device
    factorized_coeffs = []

    # Initialize lists to store coefficients for each level
    coeffs_per_level = [[] for _ in range(level + 1)]

    # Perform wavelet decomposition on each sample in the batch
    for b in range(batch_size):
        # Stack embedding dimensions as channels for multivariate decomposition
        signal = x[b].cpu().detach().numpy().T  # Shape: [embedding_dim, seq_length]

        # Perform multivariate wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level, axis=1)
        # coeffs is a list: [cA_n, cD_n, cD_{n-1}, ..., cD1]

        # Apply compression by thresholding
        compressed_coeffs = []
        for coef in coeffs:
            # coef shape: [embedding_dim, new_length]
            threshold = compression_threshold * np.max(np.abs(coef))
            coef_compressed = coef * (np.abs(coef) >= threshold)
            compressed_coeffs.append(coef_compressed)

        # Collect coefficients per level
        for lvl in range(level + 1):
            coeff_lvl = compressed_coeffs[lvl]  # Shape: [embedding_dim, new_length]
            coeffs_per_level[lvl].append(torch.tensor(coeff_lvl, dtype=torch.float32))

    # For each level, stack the coefficients and apply tensor factorization
    for lvl in range(level + 1):
        # Stack batch samples: [batch_size, embedding_dim, new_length]
        coeffs_lvl = torch.stack(coeffs_per_level[lvl], dim=0).to(device)  # [B, C, L']

        # Apply tensor factorization (e.g., low-rank approximation via SVD)
        # Reshape to [B*C, L']
        coeffs_flat = coeffs_lvl.view(batch_size * embedding_dim, -1)  # [B*C, L']
        if coeffs_flat.size(1) < 2:
            # SVD requires at least 2 columns
            # Skip factorization and pad
            coeffs_factorized = torch.zeros_like(coeffs_flat).to(device)
        else:
            try:
                U, S, V = torch.svd(coeffs_flat)
                # Keep top-r singular values
                r = min(50, S.size(0))  # Rank can be adjusted
                U_r = U[:, :r]  # [B*C, r]
                S_r = S[:r]  # [r]
                V_r = V[:, :r]  # [L', r]

                # Reconstruct the factorized coefficients
                coeffs_factorized = (
                    torch.matmul(U_r, torch.diag(S_r)) @ V_r.t()
                )  # [B*C, L']
            except RuntimeError:
                # In case SVD does not converge, fallback to original coefficients
                coeffs_factorized = coeffs_flat

        # Reshape back to [B, C, L']
        coeffs_factorized = coeffs_factorized.view(
            batch_size, embedding_dim, -1
        )  # [B, C, L']

        # Upsample to original sequence length using linear interpolation
        coeffs_factorized = F.interpolate(
            coeffs_factorized, size=seq_length, mode="linear", align_corners=False
        )  # [B, C, L]

        factorized_coeffs.append(coeffs_factorized)  # List of [B, C, L] per level

    return factorized_coeffs


class LayerWiseAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_num_layers):
        super(LayerWiseAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Initialize layer weights for current layer and previous layers
        self.max_num_layers = max_num_layers
        self.layer_weights = nn.Parameter(torch.ones(max_num_layers + 1))
        self.scaling = self.head_dim**-0.5

    def forward(self, x, layer_outputs):
        # x shape: [seq_length, batch_size, d_model]
        # layer_outputs: list of tensors with same shape as x

        seq_length, batch_size, _ = x.size()

        # Project queries, keys, and values
        q = self.q_proj(x).view(seq_length, batch_size * self.num_heads, self.head_dim)
        k = self.k_proj(x).view(seq_length, batch_size * self.num_heads, self.head_dim)
        v = self.v_proj(x).view(seq_length, batch_size * self.num_heads, self.head_dim)

        # Compute attention scores for current layer
        scores = torch.bmm(q.transpose(0, 1), k.transpose(0, 1).transpose(1, 2))
        scores = (
            scores * self.scaling
        )  # Shape: [batch_size * num_heads, seq_length, seq_length]

        # Initialize lists for all scores and values
        all_scores = [scores]
        all_values = [v]

        # Add layer-wise attention
        if layer_outputs:
            for prev_output in layer_outputs:
                k_prev = self.k_proj(prev_output).view(
                    seq_length, batch_size * self.num_heads, self.head_dim
                )
                v_prev = self.v_proj(prev_output).view(
                    seq_length, batch_size * self.num_heads, self.head_dim
                )

                score_prev = torch.bmm(
                    q.transpose(0, 1), k_prev.transpose(0, 1).transpose(1, 2)
                )
                score_prev = score_prev * self.scaling
                all_scores.append(score_prev)
                all_values.append(v_prev)

        # Stack scores and apply softmax
        stacked_scores = torch.stack(
            all_scores, dim=-1
        )  # [batch_size*num_heads, seq_length, seq_length, num_layers+1]
        attn_weights = F.softmax(stacked_scores, dim=-2)

        # Apply layer weights
        len_all_scores = len(all_scores)
        layer_weights = F.softmax(self.layer_weights[:len_all_scores], dim=0)
        attn_weights = attn_weights * layer_weights.view(1, 1, 1, -1)

        # Compute weighted sum of values
        combined_values = sum(
            attn_weights[..., i].bmm(all_values[i].transpose(0, 1))
            for i in range(len_all_scores)
        )

        # Reshape and project output
        combined_values = (
            combined_values.transpose(0, 1)
            .contiguous()
            .view(seq_length, batch_size, self.d_model)
        )
        output = self.out_proj(combined_values)
        return output


class SNNDecoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, snn_neuron, num_layers, dim_feedforward=2048, dropout=0.1
    ):
        super(SNNDecoderLayer, self).__init__()
        self.layer_wise_attn = LayerWiseAttention(d_model, nhead, num_layers)
        self.snn_neuron = snn_neuron
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, tgt, layer_outputs=None, tgt_mask=None, tgt_key_padding_mask=None
    ):
        # Apply layer-wise attention
        tgt2 = self.layer_wise_attn(tgt, layer_outputs)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Apply SNN neuron
        tgt_snn = self.snn_neuron(tgt.permute(1, 0, 2))
        tgt_snn = tgt_snn.permute(1, 0, 2)

        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt_snn))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


class SNNTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(SNNTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt
        layer_outputs = []

        for mod in self.layers:
            output = mod(
                output,
                layer_outputs=layer_outputs,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            layer_outputs.append(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


class WaveletSNNTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        snn_threshold=1.0,
        snn_decay=0.9,
        wavelet="db2",
        wavelet_level=4,
        compression_threshold=0.05,
        factor_rank=768,
    ):
        super(WaveletSNNTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.levels = wavelet_level
        self.d_model = d_model
        self.wavelet = wavelet
        self.compression_threshold = compression_threshold
        self.factor_rank = factor_rank

        snn_neuron = AdaptiveResonateAndFireNeuron(
            d_model, d_model, threshold=snn_threshold, decay=snn_decay
        )
        decoder_layer = SNNDecoderLayer(d_model, nhead, snn_neuron, num_layers)
        self.decoder = SNNTransformerDecoder(decoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)

        wavelet_coeffs = shared_wavelet_decompose(
            src,
            level=self.levels - 1,
            wavelet=self.wavelet,
            compression_threshold=self.compression_threshold,
        )

        outputs = []
        for coeffs in wavelet_coeffs:
            coeffs = coeffs.permute(2, 0, 1)
            output = self.decoder(
                coeffs, tgt_mask=src_mask, tgt_key_padding_mask=src_key_padding_mask
            )
            outputs.append(output)

        combined_output = sum(outputs) / len(outputs)
        combined_output = self.output_layer(combined_output)

        return combined_output.permute(1, 0, 2)


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# Hyperparameters
vocab_size = tokenizer.vocab_size
d_model = 1024
nhead = 8
num_layers = 16
snn_threshold = 1.0
snn_decay = 0.9


# Instantiate the model
model = WaveletSNNTransformerModel(
    vocab_size, d_model, nhead, num_layers, snn_threshold, snn_decay
)


def initialize_model_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            # Initialize the in_proj_weight and out_proj.weight
            torch.nn.init.xavier_uniform_(module.in_proj_weight)
            torch.nn.init.xavier_uniform_(module.out_proj.weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


initialize_model_weights(model)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# Load Wikitext-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# GPT-2 tokenizer does not have a padding token by default
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# Group the tokenized texts into chunks of block_size
block_size = 50  # Adjust as needed


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    # We drop the last chunk if it's smaller than block_size
    total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size
    result = {
        k: [
            concatenated_examples[k][i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        for k in concatenated_examples.keys()
    }
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
)


# Define the collate function to prepare batches
def collate_fn(examples):
    input_ids = torch.tensor([example["input_ids"][:-1] for example in examples])
    labels = torch.tensor([example["input_ids"][1:] for example in examples])
    return input_ids, labels


train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

train_dataloader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
)
eval_dataloader = DataLoader(eval_dataset, batch_size=64, collate_fn=collate_fn)


# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Use CrossEntropyLoss for language modeling
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=800, eta_min=1e-6
)
test_sentence = "He scraped his feet against the "
test_input_ids = tokenizer.encode(test_sentence, return_tensors="pt").to(device)

num_epochs = 16  # Adjust as needed
from tqdm import tqdm

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (input_ids, labels) in enumerate(tqdm(train_dataloader)):
        input_ids = input_ids.to(device)  # Shape: [batch_size, seq_length]
        labels = labels.to(device)  # Shape: [batch_size, seq_length]

        # Forward pass
        outputs = model(input_ids)
        # outputs shape: [batch_size, seq_length, vocab_size]

        # Compute loss
        loss = criterion(outputs.reshape(-1, vocab_size), labels.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")

    # --- Testing the model on a specific sentence ---
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
        predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indices.cpu().numpy())

        # Print the results
        print(f"\nTest Sentence: '{test_sentence}'")
        print("Top predictions for the next token:")
        for i in range(top_k):
            token = predicted_tokens[i]
            prob = top_k_probs[i].item()
            print(f"  {i+1}: {token} (probability: {prob:.4f})")
    model.train()  # Set model back to training mode

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")

import pickle


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


save_model(model, optimizer, epoch=0)
model.eval()
total_eval_loss = 0
with torch.no_grad():
    for input_ids, labels in eval_dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        total_eval_loss += loss.item()

avg_eval_loss = total_eval_loss / len(eval_dataloader)
print(f"Validation Loss: {avg_eval_loss}")

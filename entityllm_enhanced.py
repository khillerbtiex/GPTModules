import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List
from grokfast import gradfilter_ma, gradfilter_ema
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
import pickle, os

# Load the wikitext dataset

ds = load_dataset("abideen/Cosmopedia-100k-pretrain", split="train[:18%]")
train_texts = ds["text"]


# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token


class CreativeWritingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.input_ids = []
        self.attn_masks = []

        for text in texts:
            encodings_dict = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            self.input_ids.append(encodings_dict["input_ids"])
            self.attn_masks.append(encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx].squeeze(),
            "attention_mask": self.attn_masks[idx].squeeze(),
        }


# Create the dataset and dataloader
train_dataset = CreativeWritingDataset(train_texts, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
embed_model = GPT2Model.from_pretrained("openai-community/gpt2")
from tqdm import tqdm

# Get embeddings for unique tokens
with torch.no_grad():
    token_embeddings = embed_model.wte.weight.cpu().numpy()


def find_optimal_k(embeddings, k_range, early_stop_rounds=3):
    """Determine optimal number of clusters (k) using silhouette score."""
    best_k = None
    best_score = -1
    no_improve_counter = 0
    first_improve = 0

    for k in tqdm(k_range):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(embeddings)

        # Check that we have at least 2 clusters for silhouette score
        if len(np.unique(labels)) > 1:
            score = silhouette_score(embeddings, labels, metric="cosine")
            if score > best_score:
                best_score = score
                no_improve_counter = 0  # Reset when improvement occurs
            else:
                if no_improve_counter == 0:
                    first_improve = k  # Save first improvement
                no_improve_counter += 1
                if no_improve_counter >= early_stop_rounds:
                    return first_improve  # Stop if no improvement

    return k


def hierarchical_clustering(embeddings, max_depth=3, k_range=range(2, 32)):
    """Create a 3-level hierarchical cluster tree with optimal k at each level."""
    cluster_tree = []

    def cluster_level(embeddings, depth):
        if depth > max_depth:
            return None

        # Find optimal number of clusters for current depth
        k = find_optimal_k(embeddings, k_range)
        aggcluster = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="complete"
        )
        labels = aggcluster.fit_predict(embeddings)

        clusters = {}
        for i in range(k):
            indices = np.where(labels == i)[0]
            cluster_token_ids = indices
            cluster_embeddings = embeddings

            # Recursively cluster sub-clusters up to max_depth
            subclusters = cluster_level(cluster_embeddings, depth + 1)
            clusters[i] = {
                "token_ids": cluster_token_ids,
                "embeddings": cluster_embeddings,
                "subclusters": subclusters,
            }

        return clusters

    # Start clustering at the top level
    cluster_tree = cluster_level(embeddings, depth=1)
    return cluster_tree


def rec_tensor_fix(part_tree: dict) -> dict:
    for key, value in part_tree.items():
        if key == "subclusters":
            part_tree[key] = rec_tensor_fix(value)
        elif key in ("token_ids", "embeddings"):
            part_tree[key] = torch.tensor(value, device="cuda")
    return part_tree


def load_cluster_tree():
    file_path = "cluster_tree.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            cluster_tree = pickle.load(f)
    else:
        cluster_tree = hierarchical_clustering(token_embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(cluster_tree, f)

    print("Created cluster tree")
    return [rec_tensor_fix(cluster_tree[el]) for el in cluster_tree]


def compute_cluster_centroids(cluster_tree):
    centroids = {}  # Changed to a dictionary to use id as key

    def compute_centroid(cluster):
        # Convert embeddings to torch tensor if it's not already
        embeddings = (
            torch.tensor(cluster["embeddings"], dtype=torch.float32)
            if not torch.is_tensor(cluster["embeddings"])
            else cluster["embeddings"]
        )

        # Ensure embeddings is 2D
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        # Compute centroid
        centroid = torch.mean(embeddings, dim=0)
        cluster["centroid"] = centroid
        centroids[id(cluster)] = centroid

        # Recursively compute centroids for subclusters
        if cluster["subclusters"] is not None:
            for subcluster in cluster["subclusters"].values():
                compute_centroid(subcluster)

    # Compute centroids for each top-level cluster
    for cluster in cluster_tree:
        compute_centroid(cluster)

    return centroids


cluster_tree = load_cluster_tree()
cluster_centroids = compute_cluster_centroids(cluster_tree)
print("Created clusters and centroids")


class RelevantTokensFromCluster(nn.Module):
    def __init__(self, cluster_tree, hidden_size, model_name):
        super(RelevantTokensFromCluster, self).__init__()
        self.hidden_size = hidden_size
        self.cluster_tree = cluster_tree
        self.levels = self._extract_levels(cluster_tree)
        self.level_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_size, num_heads=8, batch_first=True
                )
                for _ in range(len(self.levels))
            ]
        )

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens(
            {"pad_token": "[PAD]"}
        )  # Ensure pad token exists
        self.token_embeddings = GPT2Model.from_pretrained(model_name).wte.weight

        # Neural networks to produce attention weights at each level
        self.level_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                )
                for _ in range(len(self.levels))
            ]
        )

        self.token_selectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 2, hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 2, self.tokenizer.vocab_size),
                )
                for _ in range(len(self.levels))
            ]
        )
        self.ffn_final_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.final_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def _extract_levels(self, cluster_tree):
        levels = []
        current_level = [cluster_tree]
        while current_level:
            levels.append(current_level)
            next_level = []
            for cluster_dict in current_level:
                for cluster in cluster_dict:
                    if "subclusters" in cluster:
                        next_level.extend(cluster["subclusters"].values())
            current_level = next_level
        return levels

    def _get_cluster_embeddings(self, clusters):
        centroids = []
        for cluster in clusters:
            # Assume centroid is precomputed and stored in cluster dictionary
            centroid = torch.tensor(cluster["centroid"], dtype=torch.float32)
            centroids.append(centroid)
        return torch.stack(centroids, dim=0)  # [num_clusters, hidden_size]

    def _get_selected_tokens(self, hidden_states, selected_tokens, level_idx):
        selected_token_ids = list(set(selected_tokens))
        selected_token_ids = torch.tensor(selected_token_ids, dtype=torch.long).to(
            "cuda"
        )

        # Compute token scores using the token selector network
        token_scores = self.token_selectors[level_idx](
            hidden_states
        )  # [batch_size, seq_length, vocab_size]

        # Extract scores for the selected tokens
        selected_token_scores = token_scores[
            :, :, selected_token_ids
        ]  # [batch_size, seq_length, num_selected_tokens]

        # Decide on the number of tokens to select
        k = 4  # e.g., k=5 for top 5 tokens

        # Select top-k tokens based on the scores
        top_token_scores, top_token_indices = torch.topk(
            selected_token_scores, k, dim=2
        )

        # Get the actual token IDs of the top tokens
        final_selected_token_ids = selected_token_ids[
            top_token_indices
        ]  # [batch_size, seq_length, k]

        # Retrieve embeddings for the final selected tokens
        final_selected_token_embeddings = self.token_embeddings[
            final_selected_token_ids
        ]
        return final_selected_token_embeddings

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_length, hidden_size = hidden_states.size()
        device = hidden_states.device
        selected_tokens = None
        final_attn_outputs = [hidden_states]
        clusters = self.cluster_tree
        # Start from the top level # Start with the root of the cluster tree
        for level_idx in self.levels:
            if not clusters:
                break

            # Get cluster embeddings for the current level
            cluster_embeddings = [
                torch.cat(self.cluster_tree[i]["embeddings"]).mean(dim=-1)
                for i in range(len(clusters))
            ]

            # Prepare queries for attention
            query = self.level_networks[level_idx](
                hidden_states
            )  # [batch_size, seq_length, hidden_size]

            # Compute attention scores
            attn_output, attn_weights = self.level_attentions[level_idx](
                query,
                cluster_embeddings.unsqueeze(0).expand(batch_size, -1, -1),
                cluster_embeddings.unsqueeze(0).expand(batch_size, -1, -1),
            )  # attn_output: [batch_size, seq_length, hidden_size]

            # Select the cluster with the highest attention
            attn_weights_mean = attn_weights.mean(dim=1)  # [batch_size, num_clusters]
            top_cluster_values, top_cluster_indices = torch.topk(
                attn_weights_mean, k=1, dim=1
            )

            # Update clusters for the next level
            new_clusters = {}
            selected_tokens = []  # Initialize if not already set
            for b in range(batch_size):
                cluster_keys = list(clusters.keys())
                for idx in top_cluster_indices[b]:

                    cluster_idx = idx.item()
                    selected_cluster = clusters[cluster_keys[cluster_idx]]
                    if selected_cluster["subclusters"] is not None:
                        new_clusters.update(selected_cluster["subclusters"])
                    else:
                        selected_tokens.extend(
                            self._get_cluster_embeddings(
                                hidden_states, selected_cluster["token_ids"], level_idx
                            )
                        )
            clusters = new_clusters  # Move to the next level

            # Compute final attention between hidden states and selected token embeddings
            if selected_tokens:
                stacked_selected_tokens = torch.stack(selected_tokens)
                final_attn_output, _ = self.final_attention(
                    hidden_states,
                    stacked_selected_tokens,
                    stacked_selected_tokens,
                )
                final_attn_outputs.extend(self.ffn_final_attention(final_attn_output))
        final_attn_outputs = torch.cat(final_attn_outputs, dim=0)
        return self.output_layer(final_attn_outputs)


class DivergenceConvergenceModule(nn.Module):

    def __init__(
        self, model_name, hidden_size, num_heads, cluster_tree, aux_memory_size=8
    ):
        super(DivergenceConvergenceModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.aux_memory_size = aux_memory_size
        self.relevant_tokens_module = RelevantTokensFromCluster(
            cluster_tree, hidden_size, model_name
        )

        # Metaphor Leg Identification Submodule
        self.leg_identification_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.leg_identification_attention_2 = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.ffn_flip_states = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Feed-Forward Network for processing legs
        self.ffn_leg_identification = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.ffn_leg_identification_2 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Random Projection
        self.random_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size)
        )

        # Contextual Embedding Pooling
        self.contextual_pooling = nn.AdaptiveAvgPool1d(1)

        # Exploration Head
        self.exploration_head = nn.Linear(hidden_size, hidden_size)

        # Auxiliary Memory for distant topics
        self.auxiliary_memory = nn.Parameter(
            torch.randn(10, hidden_size)
        )  # 10 prototypical embeddings

        # Linking Attention Mechanism
        self.linking_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )

        # FFN for linking
        self.ffn_linking = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Relationship Embeddings
        self.relationship_embeddings = nn.Embedding(
            num_embeddings=12, embedding_dim=hidden_size
        )

        # Convergence Attention Submodule
        self.convergence_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )

        # FFN for convergence
        self.ffn_convergence = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Insight Filtering
        self.insight_filtering = nn.Linear(hidden_size, hidden_size)

        # Divergent-Sequential Re-Scoring Mechanism
        self.re_scoring = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.re_scoring_2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def generate_auxiliary_memory(self, hidden_states):
        """Dynamically generates auxiliary memory based on the current batch of hidden states."""
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Sample or cluster hidden states to create auxiliary memory
        indices = torch.randint(
            0, seq_length, (self.aux_memory_size,), device=hidden_states.device
        )
        auxiliary_memory = hidden_states[
            :, indices, :
        ]  # (batch_size, aux_memory_size, hidden_size)

        return auxiliary_memory

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Phase 1: Metaphor Creation
        # 1. Metaphor Leg Identification (Divergence)
        flip_vals = (
            torch.randint(0, 2, hidden_states.shape, device=hidden_states.device) * 2
            - 1
        )
        flip_states = self.relevant_tokens_module(hidden_states, attention_mask)
        flip_states = self.ffn_flip_states(flip_states)

        legs, _ = self.leg_identification_attention(
            hidden_states,
            flip_states,
            flip_states,
            key_padding_mask=~attention_mask.bool(),
        )
        legs = self.ffn_leg_identification(legs)
        legs_pooled = self.contextual_pooling(legs.transpose(1, 2)).squeeze(-1)
        legs_expanded = legs_pooled.unsqueeze(1).expand(-1, seq_length, -1)
        exploration = torch.tanh(self.exploration_head(legs_expanded))
        combined_hidden = hidden_states + exploration

        # Contextual Shift for Distant Attention
        shifted_hidden_states = (
            combined_hidden + torch.randn_like(combined_hidden) * 0.05
        )
        shifted_hidden_states = self.random_projection(shifted_hidden_states)
        shifted_legs, _ = self.leg_identification_attention_2(
            shifted_hidden_states,
            legs,
            legs,
            key_padding_mask=~attention_mask.bool(),
        )
        shifted_legs = self.ffn_leg_identification_2(shifted_legs)
        combined_hidden += shifted_legs

        auxiliary_memory = self.generate_auxiliary_memory(
            hidden_states
        )  # Shape: (batch_size, aux_memory_size, hidden_size)

        # Cross-attention with Auxiliary Memory
        auxiliary_attention, _ = self.linking_attention(
            combined_hidden, auxiliary_memory, auxiliary_memory
        )
        combined_hidden += auxiliary_attention

        # 2. Leg Linking (Preliminary Convergence)
        linked_legs, _ = self.linking_attention(
            shifted_legs,
            shifted_legs,
            shifted_legs,
            key_padding_mask=~attention_mask.bool(),
        )
        linked_legs = self.ffn_linking(linked_legs)

        relationship_scores = torch.randint(
            0, 10, (batch_size, seq_length), device=hidden_states.device
        )
        relationship_embeds = self.relationship_embeddings(relationship_scores)
        linked_legs += relationship_embeds

        # Phase 2: Insight Extraction
        converged, _ = self.convergence_attention(
            combined_hidden,
            linked_legs,
            linked_legs,
            key_padding_mask=~attention_mask.bool(),
        )
        converged = self.ffn_convergence(converged)
        insights = torch.tanh(self.insight_filtering(converged))

        # Divergent-Sequential Re-Scoring Mechanism
        combined = torch.cat([hidden_states, insights], dim=-1)
        scores = self.re_scoring(combined).squeeze(-1)
        scored_insights = insights * scores.unsqueeze(-1)
        scores_hidden = self.re_scoring_2(
            torch.cat([hidden_states, combined_hidden], dim=-1)
        ).squeeze(-1)
        scored_hidden = combined_hidden * scores_hidden.unsqueeze(-1)

        # Final output
        output = self.output_layer(
            torch.cat([hidden_states, scored_hidden, scored_insights], dim=-1)
        )

        return output


class GPT2WithDivergenceConvergence(nn.Module):
    def __init__(
        self,
        cluster_tree,
        model_name="gpt2",
        divergence_hidden_size=768,
        num_heads=12,
    ):
        super(GPT2WithDivergenceConvergence, self).__init__()
        # Load pre-trained GPT-2 model
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.gpt2.resize_token_embeddings(len(tokenizer))

        # Define the Divergence-Convergence Module
        self.dc_modules = nn.ModuleList(
            [
                DivergenceConvergenceModule(
                    model_name=model_name,
                    hidden_size=divergence_hidden_size,
                    num_heads=num_heads,
                    cluster_tree=cluster_tree,
                ),
                DivergenceConvergenceModule(
                    model_name=model_name,
                    hidden_size=divergence_hidden_size,
                    num_heads=num_heads,
                    cluster_tree=cluster_tree,
                ),
            ]
        )

        # Define the LM Head
        self.lm_head = nn.Linear(
            divergence_hidden_size, self.gpt2.config.vocab_size, bias=False
        )

        # Ensure the hidden sizes match
        assert (
            self.gpt2.config.hidden_size == divergence_hidden_size
        ), "Hidden sizes must match."

    def forward(self, input_ids, attention_mask):
        # Get hidden states from GPT-2
        hidden_states = self.gpt2(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # [batch_size, seq_length, hidden_size]

        # Pass through Divergence-Convergence Module
        for module in self.dc_modules:
            hidden_states = module(hidden_states, attention_mask)
        # [batch_size, seq_length, hidden_size]

        # Pass through LM Head for next-token prediction
        logits = self.lm_head(hidden_states)  # [batch_size, seq_length, vocab_size]

        return logits


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


# Initialize the model
model = GPT2WithDivergenceConvergence(
    cluster_tree=cluster_tree,
    model_name="openai-community/gpt2",
)
total_params = sum(p.numel() for p in model.parameters())

print(f"Total model parameters: {total_params}")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
num_warmup_steps = 500
# Define optimizer and scheduler
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
)
num_epochs = 4
total_steps = len(train_dataloader) * num_epochs
total_steps = int(total_steps * (2 / 3))
total_steps -= num_warmup_steps

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=1e-6)

# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


grads = None
grads2 = None
# Training loop
try:
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        average_loss = 0.0
        for idx, batch in enumerate(tqdm(train_dataloader)):
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
            grads2 = gradfilter_ema(model, grads=grads2, alpha=0.98, lamb=2)
            optimizer.step()
            if idx + 1 > num_warmup_steps:
                scheduler.step()
            average_loss += loss.item()
            remainder = (idx + 1) % 50
            if remainder == 0:
                print(
                    f"\nBatch Idx {idx + 1} completed. Average Loss: {average_loss/(idx+1):.4f}"
                )
        average_loss = average_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {average_loss:.4f}")
        save_model(
            model,
            optimizer,
            epoch + 1,
            path=f"models/base_model_checkpoint_{epoch+1}.pkl",
        )
except KeyboardInterrupt:
    print("Training interrupted by user")
    save_model(
        model, optimizer, epoch + 1, path=f"models/base_model_checkpoint_{epoch+1}.pkl"
    )
except Exception as e:
    print(e)
for layer in model.gpt2.h[:8]:
    for param in layer.parameters():
        param.requires_grad = False
for module in model.dc_modules:
    for layer in [
        module.leg_identification_attention,
        module.linking_attention,
        module.convergence_attention,
    ]:
        for param in layer.parameters():
            param.requires_grad = False
for module in model.dc_modules:
    for layer in [
        module.ffn_flip_states,
        module.ffn_leg_identification,
        module.ffn_linking,
        module.ffn_convergence,
    ]:
        for i, param in enumerate(layer.parameters()):
            if i <= 3:
                param.requires_grad = False


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
prompt = "In the realm of creativity, it is the art of the imagination that drives forward humanities boundaries."
generated = generate_text(model, tokenizer, prompt, max_length=50)
print(generated)


# Load dataset
ds = load_dataset("anthracite-org/nopm_claude_writing_fixed")
train_texts = ds["train"]["conversations"]

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
tune_dataloader = DataLoader(tokenized_texts, batch_size=8)
grads = None
# Training loop
model.train()
for epoch in tqdm(range(6)):
    for batch in tqdm(tune_dataloader):
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
        grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
        optimizer.step()

    print(f"Epoch {epoch + 1} completed. Average Loss: {loss.item():.4f}")
save_model(model, optimizer, 0, path="models/model_checkpoint.pkl")

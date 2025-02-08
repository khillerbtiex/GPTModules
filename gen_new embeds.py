from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
from datasets import load_dataset
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm


class EmbeddingProcessor:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.add_special_tokens({"pad_token": "[UNK]"})
        self.tokenizer.padding_side = "left"
        self.model = GPT2Model.from_pretrained("openai-community/gpt2")
        self.model.eval()
        self.wte = self.model.get_input_embeddings().weight.detach().numpy()
        # Update vocab_keys to only include valid indices
        self.vocab_keys = list(
            range(0, min(len(self.tokenizer), self.model.config.vocab_size))
        )

    def get_contextual_embeddings(self, sentence):
        # Truncate long sequences to prevent memory issues
        max_length = self.model.config.max_position_embeddings
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.last_hidden_state.squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
        return tokens, hidden_states

    def process_sentences(self, sentences):
        contextual_embeddings = {token: [] for token in self.vocab_keys}

        for sentence in tqdm(sentences):
            if not isinstance(sentence, str) or not sentence.strip():
                continue

            tokens, hidden_states = self.get_contextual_embeddings(sentence)
            for piece, state in zip(tokens, hidden_states):
                token_id = self.tokenizer.convert_tokens_to_ids(piece)
                if token_id in self.vocab_keys:
                    contextual_embeddings[token_id].append(state.numpy())

        return contextual_embeddings

    def aggregate_embeddings(self, contextual_embeddings):
        aggregated = {}
        for token, states in tqdm(contextual_embeddings.items()):
            if states:  # Only process tokens that have embeddings
                mean_state = np.mean(states, axis=0)
                aggregated[token] = mean_state
            else:
                aggregated[token] = np.zeros(self.model.config.hidden_size)
        return aggregated

    def create_blended_embeddings(self, aggregated_embeddings):
        blended = {}
        for token in tqdm(self.vocab_keys):
            if token < len(self.wte):  # Ensure token is within valid range
                wte_vec = self.wte[token]
                contextual_vec = aggregated_embeddings[token]
                blended[token] = np.concatenate([wte_vec, contextual_vec])
        return blended

    def reduce_dimensions(self, blended_embeddings, n_components=2):
        if not blended_embeddings:
            raise ValueError("No embeddings to reduce dimensions")

        embedding_matrix = np.array(list(blended_embeddings.values()))
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(embedding_matrix)

    def process_dataset(self, dataset_name, split_ratio="train[:18%]"):
        # Load and process dataset
        dataset = load_dataset(dataset_name, split=split_ratio)
        sentences = dataset["text"]

        # Process embeddings pipeline
        print("Processing sentences...")
        contextual_embeddings = self.process_sentences(sentences)

        print("Aggregating embeddings...")
        aggregated_embeddings = self.aggregate_embeddings(contextual_embeddings)

        print("Creating blended embeddings...")
        blended_embeddings = self.create_blended_embeddings(aggregated_embeddings)

        print("Reducing dimensions...")
        reduced_embeddings = self.reduce_dimensions(blended_embeddings)

        return reduced_embeddings, blended_embeddings

    def cluster_embeddings(self, reduced_embeddings, n_clusters=5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(reduced_embeddings)
        return labels, kmeans.cluster_centers_

    def plot_embeddings(self, reduced_embeddings, n_clusters, labels):

        plt.figure(figsize=(10, 8))
        for i in range(n_clusters):
            plt.scatter(
                reduced_embeddings[labels == i, 0],
                reduced_embeddings[labels == i, 1],
                label=f"Cluster {i}",
            )
        plt.legend()
        plt.title("Blended Token Embedding Clusters")
        plt.show()


def main():
    print("Initializing embedding processor...")
    processor = EmbeddingProcessor()
    print("Processing dataset...")
    reduced_embeddings, blended_embeddings = processor.process_dataset(
        "abideen/Cosmopedia-100k-pretrain"
    )
    print("Processing complete!")
    return reduced_embeddings, blended_embeddings


if __name__ == "__main__":
    reduced_embeddings, blended_embeddings = main()

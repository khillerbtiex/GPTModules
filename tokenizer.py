# script1_train_tokenizer_embeddings.py

import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm

# Sample data: Replace with your text corpus
from datasets import load_dataset

data = load_dataset("openwebtext", trust_remote_code=True)["train"]["text"]
print("Data Loaded!")


def train_sentencepiece_tokenizer(data, model_prefix="spm", vocab_size=30000):
    with open("corpus.txt", "w") as f:
        for line in data:
            if isinstance(line, str):
                f.write(line.strip() + "\n")

    spm.SentencePieceTrainer.train(
        input="corpus.txt", model_prefix=model_prefix, vocab_size=vocab_size
    )


print("SentencePiece Tokenizer Trained!")
# Train SentencePiece model
train_sentencepiece_tokenizer(data)

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file="spm.model")


# Tokenizer Function using SentencePiece
def tokenizer(sentence):
    return sp.encode(sentence, out_type=int)


# Dataset Class
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = tokenizer(self.data[idx])
        return torch.tensor(self.vocab(tokens), dtype=torch.long)


# Vocabulary Builder
def yield_tokens(data):
    for sentence in data:
        yield tokenizer(sentence)


vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Dataset and DataLoader
dataset = TextDataset(data, vocab)
dataloader = DataLoader(
    dataset, batch_size=2, collate_fn=lambda x: pad_sequence(x, batch_first=True)
)


# Embedding Model
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)


# Training Function
def train_embedding_model(dataloader, embedding_model, epochs=10, lr=0.001):
    optimizer = optim.Adam(embedding_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = embedding_model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss}")


# Initialize and Train Embedding Model
embedding_dim = 8192
vocab_size = len(vocab)
embedding_model = EmbeddingModel(vocab_size, embedding_dim)

train_embedding_model(dataloader, embedding_model)

# Save trained embeddings
torch.save(embedding_model.state_dict(), "embedding_model.pth")
torch.save(vocab, "vocab.pth")

print("Tokenizer and Embeddings Trained and Saved!")

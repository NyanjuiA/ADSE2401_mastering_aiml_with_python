# Python script to demonstrate Federated Learning for keyboard Next-Word Prediction
# NB: We're using Pytorch as (TFF,syft and flwr) had not been updated for Python 3.12 & above (pip install torch)
# The script simulates federated learning across multiple users (Mueni, Ciku, Kamau & Bob) using
# synthetic data. It includes training metrics and visualisation.

# --------------------------------------------------------------------------------
# 0. Import the required modules
# --------------------------------------------------------------------------------
import copy
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict

# --------------------------------------------------------------------------------
# 1. Generate a synthetic dataset
# --------------------------------------------------------------------------------
def generate_sentences(base_phrases, variations=80):

    sentences = []
    fillers = ["please", "today", "now", "later", "quickly", "kindly"]
    endings = ["", "please", "now", "thanks", "ok", "lol"]

    for _ in range(variations):
        phrase = random.choice(base_phrases)
        words = phrase.split()

        if random.random() > 0.5:
            pos = random.randint(0, len(words) - 1)
            words.insert(pos, random.choice(fillers))

        words.append(random.choice(endings))
        sentences.append(' '.join(words).strip())

    return sentences

# Base phrases for each user(device)
mueni_base = [
    "hello how are you",
    "how is your day",
    "are you coming today",
    "let us meet later"
    "please call me"
]

ciku_base = [
    "hi how are things",
    "are you doing well",
    "what are you doing",
    "let us catch up"
    "text me later"
]

kamau_base = [
    "hello are you okay",
    "how have you been",
    "uko aje leo",
    "tutaonana later"
    "niko sawa"
]

bob_base = [
    "yo",
    "what's new",
    "hello friend",
    "how do you do"
    "goodbye"
]

# Generate the dataset
data:Dict[str,list[str]] = {
    "Mueni": generate_sentences(mueni_base),
    "Ciku": generate_sentences(ciku_base),
    "Kamau": generate_sentences(kamau_base),
    "BoB": generate_sentences(bob_base)
}

# --------------------------------------------------------------------------------
# 2. Vocabulary
# --------------------------------------------------------------------------------
def build_vocab(sentences: List[str]) -> Dict[str, int]:

    words = []
    for s in sentences:
        words.extend(s.split())

    vocab = {w:n + 1 for n, w in enumerate(set(words))} # use set to filter duplicates
    vocab["<PAD>"] = 0
    return vocab

all_sentences = sum(data.values(), [])
vocab = build_vocab(all_sentences)
vocab_size = len(vocab)

# --------------------------------------------------------------------------------
# 3. Dataset
# --------------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, sentences, vocab, seq_len=3):
        self.data = []

        for sentence in sentences:
            tokens = [vocab[w] for w in sentence.split()]
            for n in range(len(tokens) - seq_len):
                self.data.append(tokens[n: n + seq_len],tokens[n + seq_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

# --------------------------------------------------------------------------------
# 4. The Model
# --------------------------------------------------------------------------------
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=16, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --------------------------------------------------------------------------------
# 5. Training & Evaluation
# --------------------------------------------------------------------------------
def train(model, loader, epoch = 1):

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(epoch):
        for x, y in loader:
            optimiser.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimiser.step()

def evaluate_loss(model, loader):

    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0

    with torch.no_grad():
        for x, y in loader:
            loss += criterion(model(x), y).item()
    return loss / len(loader)

def evaluate_accuracy(model, loader):

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    return correct / total

# --------------------------------------------------------------------------------
# 6. Federated Averaging
# --------------------------------------------------------------------------------
def federated_average(models):

    global_model = copy.deepcopy(models[0])
    global_dict = global_model.state_dict()

    for key in global_dict:
        global_dict[key] = torch.stack(
            [m.state_dict()[key].float() for m in models]
        ).mean(0)

    global_model.load_state_dict(global_dict)
    return global_model

# --------------------------------------------------------------------------------
# 7. DataLoaders
# --------------------------------------------------------------------------------
client_loaders = {
    name: DataLoader(TextDataset(sentences, vocab),batch_size=4, shuffle=True)
    for name, sentences in data.items()
}

# --------------------------------------------------------------------------------
# 8. Federated Training
# --------------------------------------------------------------------------------











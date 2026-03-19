import time

# ----------- not in video
from collections import Counter
from math import log2

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------

# hyper param
batch_size = 32  # how many independant sequence will be process in parallel
block_size = 8  # what is the max context len for prodeiction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
# --------

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# unique char in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create mapping for characters to ins
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    encoded = []
    for c in s:
        encoded.append(stoi[c])
    return encoded


# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of ints
def decode(encoded: list[int]) -> str:
    decoded = ""
    for i in encoded:
        decoded += itos[i]
    return decoded


# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 90%
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # tells pytoch we ll not do backprop on those data
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token
        # self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )  # short for number of embedding dimensions
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # short for languagemodeling head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of ints
        tok_emb = self.token_embedding_table(idx)  # B, T, C
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = tok_emb + pos_emb
        logits = self.lm_head(x)  # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the cirrent context
        for _ in range(max_new_tokens):
            # get the prediction
            #
            # ------------ quick inermediate fix
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            #
            # logits, loss = self(idx)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becoms (B, C)
            # apply softmax
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1
        return idx


model = BigramLanguageModel()
m = model.to(device)

# create a Pytorch optimizer
start = time.time()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(f"Trained for {time.time() - start} seconds")

# ------------- not in the video
# little experiment


def shannon_entropy_text(text: str) -> float:
    if not text:
        return 0.0

    counts = Counter(text)
    total = len(text)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * log2(p)

    return entropy


# ----------------------------------

context = torch.zeros((1, 1), dtype=torch.long, device=device)
for iter in range(max_iters):
    # every once in a while evalutate the loss on train and val sets
    if iter % eval_interval == 0 or iter == 2999:
        losses = estimate_loss()
        print(
            f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    if iter == 0:
        print("----First gen---------")
        print("At step 0, after a new line the 500 chars the model generates are:")
        text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
        print(text)
        print("------ENTROPY---------")
        entropy = shannon_entropy_text(text)
        print(f"Shannon Entropy is : {entropy}")
        print("----------------------")

    # sample batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("-------------")
print(
    f"After step {max_iters}, after a new line the 500 chars the model generates are:"
)
text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(text)
print("-------------")
print("------ENTROPY---------")
entropy = shannon_entropy_text(text)
print(f"Shannon Entropy is : {entropy}")
print("----------------------")

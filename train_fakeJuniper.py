import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# read shakespeark text to check what is inside and if everything's working
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Numbers of chars in the text is: {len(text)}")

# check if all is ok
# print(f"exemple of the text\n {text[:1000]}")

# probing all character to see how many different type or char there is
chars = sorted(list(set(text)))
vocab_size = len(chars)

# i = 1
# for c in chars:
#     print(f"char {i}: {repr(c):>6}  U+{ord(c):04X}  bytes:{c.encode('utf-8').hex()}")
#     i += 1
print(f"List of chars:\n{''.join(chars)}")
print(f"Numbers of unique chars: {vocab_size}")


# Basic embedding where 1 char = one token basically. This is ehavy i think, but this is to explain from scratch
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    embedding = []
    for c in s:
        embedding.append(stoi[c])
    return embedding


# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of ints
def decode(embeds: list[int]) -> str:
    s = ""
    for i in embeds:
        s += itos[i]
    return s


# # decode = lambda l: "".join([itos[i] for i in l])

# print(encode("hii there"))
# print(decode(encode("hii there")))

time1 = time.time()

# encoding the entire text and store into a torch Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[:1000])
print(f"Encoding text took:{time.time() - time1} seconds")

# let's now plit up the data into train and validation set
# 90% train 10% to validate.
# I guess karpathy does this to verify the model is not overffiting?
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[: block_size + 1]

# x = train_data[:block_size]
# y = train_data[1 : block_size + 1]
# for t in range(block_size):
#     context = x[: t + 1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")


# Defininf parrallel task for the gpu in a 4*8 matrix ?

# torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# # one matrix of the "current" block
# # one with the "next token" after the block -> the target
# # this is to demonstrate us how we are going to give the disired output from input.
xb, yb = get_batch("train")
# print("inputs:")
# print(xb.shape)
# print(xb)
# print("targets:")
# print(yb.shape)
# print(yb)

# print("----")


# for b in range(batch_size):  # batch dimension
#     for t in range(block_size):  # time dimension
#         context = xb[b, : t + 1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target: {target}")

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # each int will be save in the embedding table at "int's row".
        # so 24 -> 24 row etc.
        # is this a lookup table like a db?
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of ints
        logits = self.token_embedding_table(idx)  # (B,T,C) Batch, Time, Channel; Tensor

        if targets is None:
            loss = None
        else:
            # cross entropy in pytorch docs -> wants B*T, C order 2D array ? I am lost :D
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(T * B)  # can be shorthanded with -1
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get prediction
            logits, loss = self(idx)
            # focus only on the last tine step
            logits = logits[:, -1, :]  # keep only B, C
            # softmax
            probs = F.softmax(logits, dim=1)  # B, C
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # becomes B, 1
            # append samples index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1
        return idx


# loss can be estimate like so:
# - ln(1/65) = 4.7
# Current loss is 4.8


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

batch_size = 32
startOpt = time.time()
trained_steps = 10000
for steps in range(trained_steps):
    # sample a batch of data
    xb, yb = get_batch("train")

    # evbaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(f"for {trained_steps} it took {time.time() - startOpt} seconds")
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))

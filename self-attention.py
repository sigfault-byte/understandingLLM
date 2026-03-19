import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 32  # Batch , time, channels
x = torch.randn(B, T, C)

# let's see a single Head perform self-attention
head_size = 16

key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

print(f"Key is {key}")

k = key(x)  # B T 16
q = query(x)  # B T 16
wei = q @ k.transpose(-2, -1)  # B T 16 @ (B 16 T) --> (B, T, T)


tril = torch.tril(torch.ones(T, T))
print(f"Tril: {tril}")

wei = wei.masked_fill(tril == 0, float("-inf"))
print(f'wei = wei.masked_fill(tril == 0, float("-inf")): {wei[0]}')

wei = F.softmax(wei, dim=-1)
print(f"wei[0] = F.softmax(wei, dim=-1):  {wei[0]}")

v = value(x)
out = wei @ v
print(f": out[0]: {out[0][0]}")

print(out.shape)

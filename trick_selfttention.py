import torch
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2  # Batch , time, channels
x = torch.randn(B, T, C)
x.shape

torch.Size([4, 8, 2])

# We want x[b, t] = mean {i <= t} * [b, i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)

#
print("x[0] is:")
print(x[0])
print(x[0][0])
print("xbow[0] is")
print(xbow[0])
print(xbow[0][0])

# Toy exemple
torch.manual_seed(1337)
a = torch.ones(3, 3)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print(f"a={a}")
print("---")
print(f"b={b}")
print("---")
print(f"c={c}")


xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)

#
# print("using torch.ones(3, 3) for a")
# print("x[0] is:")
# print(x[0])
# print(x[0][0])
# print("xbow[0] is")
# print(xbow[0])
# print(xbow[0][0])

# # Toy exemple
# torch.manual_seed(1337)
# a = torch.tril(torch.ones(3, 3))
# b = torch.randint(0, 10, (3, 2)).float()
# c = a @ b
# print("torch.tril(torch.ones(3, 3)) for a")
# print(f"a={a}")
# print("---")
# print(f"b={b}")
# print("---")
# print(f"c={c}")

# torch.manual_seed(1337)
# a = torch.tril(torch.ones(3, 3))
# a = a / torch.sum(a, 1, keepdim=True)
# b = torch.randint(0, 10, (3, 2)).float()
# c = a @ b
# print("torch.tril(torch.ones(3, 3)) for a")
# print("a = a / torch.sum(a, 1, keepdim=True)")
# print(f"a={a}")
# print("---")
# print(f"b={b}")
# print("---")
# print(f"c={c}")

# wei = torch.tril(torch.ones(T, T))  # -> a, the weight !
# wei = wei / wei.sum(1, keepdim=True)
# xbow2 = wei @ x  # (B , T) @ (B, T, C) ---> (B, T, C)
# print(torch.allclose(xbow, xbow2)) # are these tensors numerically almost equal?

tril = torch.tril(torch.ones(T, T))
print(f"Tril: {tril}")
wei = torch.zeros((T, T))
print(f"wei = torch.zeros: {wei}")
wei = wei.masked_fill(tril == 0, float("-inf"))
print(f'wei = wei.masked_fill(tril == 0, float("-inf")): {wei}')
wei = F.softmax(wei, dim=-1)
print(f"wei = F.softmax(wei, dim=-1):  {wei}")
xbow3 = wei @ x
print(f": xbow3: {xbow3}")
torch.allclose(xbow, xbow3)

`Tue Mar 17 10:24:42 WET 2026`

# Math trick of self-attention

Karpathy [video stamp](https://youtu.be/kCc8FmEb1nY?t=2784)

# Creating a "bad" uniform average

We are using a purposely un optimized O(N^2) loop to make a uniform average!

```python

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
```

We can see it with the simplified loop!
`x[b][t][c]` !

```
for each sequence b
  for each time t
    look at all previous i ≤ t
      aggregate values for each channel c
```

> xbow = RandomNumbers in a 3D array.
> representing our Batch ( sequences ) = 4
> the *time* Karpathy says, which are the individual tokenID in the sequence = 8
> the channels, or the embedding dimensions.

---

# torch.tril

## ouptut comparison

```bash

using torch.ones(3, 3) for a
x[0] is:
tensor([[ 0.1808, -0.0700],
        [-0.3596, -0.9152],
        [ 0.6258,  0.0255],
        [...]])
tensor([ 0.1808, -0.0700])
xbow[0] is
tensor([[ 0.1808, -0.0700],
        [-0.0894, -0.4926],
        [ 0.1490, -0.3199],
        [...]])
tensor([ 0.1808, -0.0700])


torch.tril(torch.ones(3, 3)) for a
a=tensor([[1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]])
---
b=tensor([[5., 7.],
        [2., 0.],
        [5., 3.]])
---
c=tensor([[ 5.,  7.],
        [ 7.,  7.],
        [12., 10.]])

```

Using torch.tril turns numbers in the uper right triangle inside an array to 0:

Hence during matrix multiplication, with 1s or 0s in a:

```python

c[0][0] = 1 * 5 + 0 * 2 + 0 * 5 = 5
c[0][1] = 1 * 5 + 1 * 2 + 0 * 5 = 7
c[0][2] = 1 * 5 + 1 * 2 + 1 * 5 = 12 

```
## masking with ` a/a(sum of row)`

```python

torch.tril(torch.ones(3, 3)) for a
a = a / torch.sum(a, 1, keepdim=True)
a=tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])
---
b=tensor([[5., 7.],
        [2., 0.],
        [5., 3.]])
---
c=tensor([[5.0000, 7.0000],
        [3.5000, 3.5000],
        [4.0000, 3.3333]])

```
This is neat:
```bash
[1, 0, 0] / 1 → [1, 0, 0]
[1, 1, 0] / 2 → [0.5, 0.5, 0]
[1, 1, 1] / 3 → [0.333, 0.333, 0.333]
```
This is automatically (thanks tensor in pytorch) preserves the reduced dimension as size 1 ( because keepdim=True (3,1) so a 2D array with a single colum. 

The resulting `c[2]`with triangle unnormalized mask 

```bash
c=tensor([[ 5.,  7.],
 20         [ 7.,  7.],
 21         [12., 10.]])

```

And after :

```bash
c=tensor([[5.0000, 7.0000],
        [3.5000, 3.5000],
        [4.0000, 3.3333]])

```
> the lower triangular mask is normalized row wise so each row becomes a probability distribution over the allowed past positions

> For now only the tri maks is normalized.

## last version of xbow:

```python

tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=1)
xbow3 = wei @ x

```

`Tril` -> 8\*8 tensor mask triangle similar to previous a.
`wei = torch.zero` -> all zero in a 8\*8 tensor
`wei.masked_fill` -> using the triangle mask to decide where to put -inf, 1 = 0, 0 = -inf
`F.softmax` -> softmax(x\_i) = exp(x\_i) / sum\_j exp(x\_j)
Because e^0 = 1, we can "get back" the 1 but the -infinity stay 0 !

`wei @ x` -> computes a causal weighted average of past token vectors. It is the same as before (`torch.allclose(xbow, xbow3 -> true`)  because we are in a specific case 
> `wei @ x` -> is aka **aggregation** wei contains the attention weights, and wei @ x applies them to aggregate the values

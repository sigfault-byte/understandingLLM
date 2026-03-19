`Wed Mar 18 14:07:03 WET 2026`

# Self Attention - One head

Following the [video](https://youtu.be/kCc8FmEb1nY?t=3767)

In the toy version: [[[README-20250317-toy-attention]] 
Attention was *just* a **simple average of past token**.

```python
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)


k = key(x)
q = query(x)
wei = q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
print(f'wei = wei.masked_fill(tril == 0, float("-inf")): {wei[0]}')

wei = F.softmax(wei, dim=-1)
print(f"wei = F.softmax(wei, dim=1):  {wei[0]}")

v = value(x)
out = wei @ v
print(f": out[0]: {out[0]}")
```

## What are Key, Query, Value?

`print(key)`:
```python
Key is Linear(in_features=32, out_features=16, bias=False)
```

Linear is `y = xA^T + b`.
`^T` is not *power of*, it is *Transpose*.
Meaning the dimensions **are flipped**:
```bash
	a b c         a d g
A =	d e f   A^T = b e h
	g h i         c f i
	
if shape(B)   = (3, 2)
   shape(B^T) = (2, 3)
```
 
 Whatever 32D vector argument passed to Key

---
## So what are k, q, (v)?

> Ignoring `v` for now this is another trick

`x` is (B,T,C) -> (4, 8, 32)
 Here`x[0][0]` is the 32D representation of the first token position in the first batch item.
 It is shape(32,)
 That 32D token representation is transformed by the learned linear map into a 16D query/key/value vector. 
 And returns a (16,) for `y[0][0]` ( Called **FEATURES** of Query Key Value)
 > We are compressing `x` shape(B, T, C) to `y`shape(B, T, head\_size)
 
 > FIX: It is not compressing, it is more apt to say: *each token representation is projected into three different learned subspaces* 
 > Theses subspace can have different dimensions depending on `n_embed` value and `head_size`

Same for all of q, k, v. They are random at start, but gives a playground for the model *head* to compare different dimensions that will represent different information.
> I guess that when entropy *drops* the q, k, v start to have some *meaning*.

> FIX: **as training reduces loss, the model learns projections that make useful comparisons and routing possible**

## What is wei now?

`q` is shape (B, T, head_size)
`k` is shape (B, T, head_size)
`k.transpose(-2, -1)` shape is now (B, head_size, T)
wei is shape (B, T, T) 
Its result is a scalar:
	- for each batch
	- for each query
	- for each key
	- how similar are q and k

**Each query vector is compared against every key vector through a dot product over the head dimension**:
`(B, T, head_size) @ (B, head_size, T) -> (B, T, T)`
multiply the row `T` of q -> `head_size` dimension
with the column `head_size` dimension

Multiplying filters *filters*.
> This is a *bit* like fragment colors, we do not use AND we use a multiplication. If the values are *opposed* the results cancels out, otherwise they are amplified by the product.

**For each token `i` and token `j`, compute a similarity score between `query(i)` and `key(j)`**.

`wei[b, i, j]` **is the raw compatibility score telling how much token i wants to attend to token j.**

## From  logits to probabilities

`wei = wei.masked_fill(tril == 0, float("-inf"))`
`print(wei[0])` :
```bash
[
[-1.7629, -inf   , -inf   , -inf   , -inf   , -inf   , -inf   , -inf], 
[-3.3334, -1.6556, -inf   , -inf   , -inf   , -inf   , -inf   , -inf],
[-1.0226, -1.2606, 0.0762 , -inf   , -inf   , -inf   , -inf   , -inf],
[ 0.7836, -0.8014, -0.3368, -0.8496, -inf   , -inf   , -inf   , -inf],
[-1.2566, 0.0187 , -0.7880, -1.3204, 2.0363 , -inf   , -inf   , -inf],
[-0.3126, 2.4152 , -0.1106, -0.9931, 3.3449 , -2.5229, -inf   , -inf],
[ 1.0876, 1.9652 , -0.2621, -0.3158, 0.6091 , 1.2616 , -0.5484, -inf],
[-1.8044, -0.4126, -0.8306, 0.5898 , -0.7987, -0.5856, 0.6433 , 0.6303]
]


```

Now the old *uniform average* is gone.
The mask turn all values in the upper right corner to *-infinity*.
The scores are ready, they are not uniform. 

Now applying Softmax:
```bash
[
[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], 
[0.1574, 0.8426, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], 
[0.2088, 0.1646, 0.6266, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], 
[0.5792, 0.1187, 0.1889, 0.1131, 0.0000, 0.0000, 0.0000, 0.0000], 
[0.0294, 0.1052, 0.0469, 0.0276, 0.7909, 0.0000, 0.0000, 0.0000], 
[0.0176, 0.2689, 0.0215, 0.0089, 0.6812, 0.0019, 0.0000, 0.0000], 
[0.1691, 0.4066, 0.0438, 0.0416, 0.1048, 0.2012, 0.0329, 0.0000], 
[0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]
]
```

`sum_j wei[b, i, j] = 1` -> each token distributes **100% of its attention budget** over previous tokens

**Scores are now probabilities.**

## The weighted mixture of value vectors relevant to a token 

`v = value(x)` -> Another ( B, T, head_size) matrix where the model can use another multidimensions to represent the token *current* value *for* this *specific* context.

`out = wei @ v` or `out[b, i, :] = sum_j wei[b, i, j] * v[b, j, :]`
`print(wei[0][0])`
```bash
[-0.1571, 0.8801, 0.1615, -0.7824, -0.1429, 0.7468, 0.1007, -0.5239, -0.8873, 0.1907, 0.1762, -0.5943, -0.4812, -0.4860, 0.2862, 0.5710]
```

`wei` is shape (B, T, T)
`v` is shape (B, T, head_size)
`out` is (B, T, 16)
	- for token `i`
	- take token `j`'s value vector, 
	- scale this vector by how much `i` *tends* to `j`
	- sum all of this to a new row vector with `head_size` dimensions

>**This row vector is a new embedding of *what* the token is in this context.**
>*or*
>**This row vector is a new embedding of what the  token means in this context.**

---
At this [moment](https://youtu.be/kCc8FmEb1nY?t=4911) Karpathy tweaks 
`max_iters` from 3000 to 5000
and lr from 1e^-2 to 1e^-3.

Before tweaking, i ran with the same parameters to compare to the [[README-20250317-bigram2| previous result]].

---

# Self-attention - Multi Heads

> surprisingly short code, mostly easy to understand:

```python
class MultiHeadAttention(nn.Module):
    # multiple heads of self attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

	...

class BigramLanguageModel(nn.Module):
		...
		self.sa_heads = MultiHeadAttention(4, n_embd // 4) 
		
	def forward(self, idx, Target=None):
		...
		x = tok_emb + pos_emb
	    x = self.sa_heads(x)
		...
```

In the model class the old head code:
```python
self.lm_head = nn.Linear(n_embd, vocab_size)
...
x = tok_emb + pos_emb
```

## Defining the multihead

`self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))`
> Let's not forget that **the heads are independent parallel attention pathways, all applied to the same input x.**

With *a little* help from torch, creating a `num_heads` parallel processing function is straightforward.

`return torch.cat([h(x) for h in self.heads], dim=-1)`
For the sake of clarity rewriting the python shorthand to a normal loop:
```python
outs = []
for h in self.heads:
    outs.append(h(x))   # each h(x) is (B, T, head_size)

return torch.cat(outs, dim=-1)
```

Each head is (B, T, head_size)

outs before `torch.cat `:
```bash
out = [tensor0.shape(B, T, 8),tensor1.shape(B, T, 8),tensor2.shape(B, T, 8),...]
```
> `dim =-1` means *on last axis*

`return torch.cat(outs, dim=-1)`
Concatenated the last dimension `head_size` (joining them) 
Result is the *same* `n_embed` dimension, because of how `sa_heads` is called in object BigramModel :

`self.sa_heads = MultiHeadAttention(4, n_embd // 4) `

head_size and num_heads cannot be chosen arbitrarily.
They must satisfy:
`num_heads * head_size = n_embd`

In this implementation, we set:
`head_size = n_embd // num_heads`
because:
`x` *must be* shape(B, T, C=n_embd)

---

> quick notes:

The heads do not directly communicate with each other inside the multi-head attention block.
Hence, each head can learn to focus on different kinds of information or token relationships.

This resembles the pattern seen in CNNs, where different filters learn to detect different features, and their outputs are combined later to give the next layer a richer picture.

> >The analogy is not exact:
> >CNN filters detect local spatial patterns, while attention heads learn dynamic token-to-token relationships.

But in both cases, several specialized *subcomponents* contribute different views of the same input.

Here, the outputs of the heads are concatenated, producing a new contextualized representation of `x`.
This enriched representation can then be processed by later layers.

When Karpathy says *the heads have a lot to talk about,* he is likely referring to that different heads can learn different useful patterns in context, and together they produce a richer contextual embedding than a single head.


---
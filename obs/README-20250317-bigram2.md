`Tue Mar 17 15:42:25 WET 2026`

# Little CPU wants to communicate part 2

Following the [video](https://youtu.be/kCc8FmEb1nY?t=3618) 

```python
class BigramLanguageModel(nn.Module):
	...
	self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  
       	self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x) 
	...

```

## Quick observations about token embedding and token context

The model is moving from a bigram lookup table `Embedding(vocab, vocab)` to a representation model `Embedding(vocab, n_embd)`.
Previously, each token's **parameters** directly produced **logits** through a row lookup:
	- *Which token ID is this?*
	- Given this token, *what are the scores for each possible next token*?

In that version, each row of the table could be read directly as next token scores over the vocabulary (here individual letters).

The model then sampled from the learned probability distribution over next tokens. 
> Because `torch.multinomial` is **stochastic**, it does not always choose the most probable token.

One of **Embed** definition is as follows:  
> **implant an idea, or feeling, so that it becomes ingrained within a particular context**
> *This is not the only definition but this one is interesting for the following reflection*

In the new version, a token is first mapped to a **learned vector** in embedding space (`n_embd` dimensions), then combined with **positional information**, and only afterwards **projected back** into vocabulary **logits** by lm_head.

>In that sense, the embedding space is where the model can encode richer structure than a direct bigram lookup. 
>The token is no longer only associated with **immediate next-token statistics**, but represented in a learned space that can **later support contextual reasoning**.

At this stage, the embedding space begins as noise.
Training must then act as a filtering process, gradually organizing that noise into a structure the model can use, so token identity and token position become meaningful signals.

---
## 1. in __init__ :

`self.token_embedding_table = nn.Embedding(vocab_size, n_embd) → (vocab_size, n_embd)`
> Karpathy notes shape (B, C), refers to **outputs after lookup**

`n_embd = 32` -> appeared to be randomly selected, but it is 1/2 of what we had previously (64)
Thus, token\_embedding\_table is now *only* 64\*32.

Row of all unique token (64 values) / column of number of embedding dimensions (32)

So a is **each token is associated with a learned 32D vector**.

**IT IS NOW AN EMBEDDING OF THE ID OF A GIVEN TOKEN**: the coordinates of the token in a learned vector space. 
> Mental *false* explanation : *intrinsic* representation of the token in model space 

`self.position_embedding_table = nn.Embedding(block_size, n_embd) → (block_size, n_embd)`
> Karpathy notes T, C. -> refers to **outputs after lookup** it is the temporal AKA the *position* of an index 

This is, in this code, 8\*32,

So this holds 32 parameters that depends on a token position. 

Another 32D vector, a **embedding of a token positional signal**. 
It is a **learned vector associated with a sequence position**.
> Mental *false* explaination: *extrinsics* of the token, the contextual placement signal

`self.lm_head = nn.Linear(n_embd, vocab_size)`
Linear transformation (affine) `y=xW + b`. 
Where: 
	- x → input vector (here: size 32)
	- W → weight matrix (32 × vocab\_size)
	- b → bias vector (size vocab\_size)
	- y → output vector (size vocab\_size)

> unsure I am understanding the full logic yet...
> basically: each token position representation (32D) is projected into 65 logits, one score for each possible next token in the vocabulary (Len(vocab_size))

> This is the prediction layer, not the embedding step.
> It maps the internal representation back into vocabulary logits.

**All this used to be:**
```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)`
```

## 2. in forward:

`tok_emb = self.token_embedding_table(idx)` -> 3D (B, T, C)
C now *only* 32 -> this is the **tokenIdentity embedding** !

`pos_emb = self.position_embedding_table(torch.arange(T, device=device))` -> 2D  (T, n_emb)
pos_emb -> Lookup rows in (block_size, C) -> returns T, C
So pos_emb[0] = index **0** -> 32 vector

`x = tok_emb + pos_emb`
This is *nothing* more than a ZIP over the T dimension (position axis!).
The operation is in **PARALLEL**. (!)

`pos_emb[t]` and `tok_emb[:, t]` are aligned along the position axis.

They are combined element-wise to form the final **embedding**.
> FUCKING 2 HOURS to understand this, i missed the parallelism completely !

**This is the final token representation for a token occurrence: token identity + token position, expressed as an n_embd dimensional vector in latent space.**

**All this used to be:**
```python
logits = self.token_embedding_table(idx)  # B, T, C
```


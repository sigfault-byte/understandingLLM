`start : Mon Mar 16 16:34:25 WET 2026`

# First day: little CPU wants to communicate

Following Karpathy **Let's build GPT: from scratch, in code, spelled out.**
 [link](https://www.youtube.com/watch?v=kCc8FmEb1nY]

Text source is a 1MB text file of [Shakespeare](https://github.com/karpathy/ng-video-lecture/blob/master/input.txt)

# Tokenisation

*Tokenizing is how we split text into units called **tokens**.*

extract unique characters
then sort them for deterministic vocabulary order

```python
chars = sorted(list(set(text)))
```

From this get the exact numbers of different characters in the text, and what they are.
This text has 65 differents characters.

---

# Encoding

*Encoding is how we turn those **token** into numerical representation*. 

Define basic *encoding* functions
```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

Then we **encode** them:
```python
def encode(s: str) -> list[int]:
    encoded = []
    for c in s:
        encoded.append(stoi[c])
    return encoded
```

---

# Loading the encoded token into a tensor
>The data is split into two.

A tensor is a *multidimensional array*.
It contains all the tokens IDs. 

`torch.long` are 64 bit ints
the 90% separation is to at the end test the model if it over fitted.
In the end we can measure if the model is able to **generalize** what it learned on unknown text.

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```
---

# Loading the data 

Batch logic to get the reference of the output

```python
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```
`ix` is a 1D tensor of random starting indices, one per batch element.

`x`: 2D vector of shape(`batch_size`, `block_size`) represents the TokenID in the block\_size 
`y`: 2D vector of shape(`batch_size`, `block_size`) represents the shifted ( +1) target chunks that follows TokenID from `x`
> `to(device)` is to dynamically compute either on CPU or GPU
> 2D tensor shape is (nb of rows, nb of cols)

Example with `batch_size=4` and `block_size=8`
```bash
inputs:
torch.Size([4, 8])
tensor([[24, 43, 58,  5, 57,  1, 46, 43],
        [44, 53, 56,  1, 58, 46, 39, 58],
        [52, 58,  1, 58, 46, 39, 58,  1],
        [25, 17, 27, 10,  0, 21,  1, 54]])
targets:
torch.Size([4, 8])
tensor([[43, 58,  5, 57,  1, 46, 43, 39],
        [53, 56,  1, 58, 46, 39, 58,  1],
        [58,  1, 58, 46, 39, 58,  1, 46],
        [17, 27, 10,  0, 21,  1, 54, 39]])
```

---

# BigramLanguageModel class

## nn.module and ( fakish ) Embedding

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```
`nn.module` is `torch.nn.Module`
as a bunch of methods like:
	- parameter registration
	- model hierarchy
	- .state\_dict() saving/loading
	- .to(device)
	- .train() / .eval()
	- integration with optimizers
	 
BigramLanguageModel inherits from nn.Module
Add add a learnable parameter layer: `token_embedding_table`

A 2D matrix. Each row : tokenX relationship with all other token initialize with *random values*. It is the `logits`: the relationship between tokens!

**More precisely:** each row = logits for the next token given the current token
so **row i → scores for every possible next token** 
> **IMPORTANT** : the `token_embedding_table` at this moment is the initialization of the weights !

> Usually te embedding is `(vocab_size, embedding_dimension)`

## forward pass and Cross Entropy

```python
    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)  

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

When passing `idx` to `token_embedding_table` pytorch performs a row lookup for every TokenID in idx.

Hence:
**Logits is a 3D tensor with shape (B, T, C)**
where:
	- B = batch\_size (number of sequences)
	- T = token positions in each sequence
	- C = vocabulary size:

Because now each token returns a vector that contains the logits (scores) for every possible next token. 

**More precisely** : Passing idx to `token_embedding_table` performs a lookup for every token ID.
Each token returns a vector of length `C` (`vocabulary size`), producing a tensor
of shape (B, T, C).

For each token position, the model outputs C scores, one for each possible next token in the vocabulary
> This is kinda nuts how hard it is to visualize.
> `embeding_table` is 2D. 
> `idx` is shape(4,8)
> when `embedding_table` receives `idx` it returns, for each tokenID a vector of 65 values -> shape (4,8,65)
 
> **IMPORTANT**:
> We go from a 2D tensor (B, T) to a 3D tensor (B, T, C) because each token ID in idx is used to LOOK UP a row in the embedding table (which contains the weights). We are not yet  *properly* applying the weights. But this is where *sh1t* happens. 

Each lookup returns a vector of size C (vocabulary size).

So for B\*T tokens, we obtain B\*T vectors → reshaped as (B, T, C).

This means that for each sequence, and each token in the sequence, the model
returns a vector of logits representing scores for every possible next token.

`F` comes from `from torch.nn import functional as F`
`cross_entropy` expects shapes like (N, C)
	- `N`: number of training examples
	- 'C': vocab\_size
 
So for both logits and targets, `B` and `T` are flattens. Instead of having 4row and 8 columns, we get 4\*8 row and `C` column
target is a class index, it holds the truth where the models should try to tend to.

> cross_entropy internally does:
>	- softmax(logits)
>	- pick probability of correct class
>	- compute -log(probability)

## Generate next token ( !! )

```python
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  
            probs = F.softmax(logits, dim=1)  
            idx_next = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat((idx, idx_next), dim=1)  
        return idx

```
> self(idx) is some magic shorthand from nn.Modules, it is *basically* equivalent to logits, loss = self.forward(idx).

`[:, -1, :] only grabing the last timestep ( keep only `B` and `C` -> batch size and score !)

During training the model predicts the next token for **every position** in the sequence so it can learn from many examples in parallel.

During generation **only** the prediction from **the last token** in the current sequence**, because that is the token that will **determine the next generated token**.

Softmax converts logits (model scores) into **probabilities** between 0 and 1 that sum to 1.

torch.multinomial samples a token index from the probability distribution.
> Theorically every token whose probs is > 0 has a *chance* to be generated.

> Temperature is **not** related to the sampling step. 
> It modifies the logits before softmax, which changes the probability distribution used by
multinomial sampling.

---

# Estimate / measure  loss


```python
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


```

`torch.zero(eval_iter)` returns a 1D tensor of zeros of length `eval_iters`.
`get_batch` retuns the `inputs` and `target`: 2D tensors(4;8) : (sequence, number\_of\_token)
`model(X.Y)` calls forward pass (nn.Module dunder __Call__)
`loss.items()` stores the numeric loss value into position `k` of the loss tensor
build a mean for each split (train and verification)
`model.train()` calls special function inherited from nn.Module.
> model.train() does **not train** : it puts the model in training mode
> `model.train()` and `model.eval()` are *sort of* changing the model state so it uses different characteristics. train() -> practice mode with randomness and noise, eval() -> deterministic without noise 

---

# Optimizer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
`model.parameters()` returns the weights. 
`Adam` updates the model's weights using gradients computed by backpropagation
`lr` how much the weights change in response to gradients

---

# Evaluate and backprop

```python
logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```
`optimizer.zero_grad(set_to_none=True)` resets gradients. clears previously accumulated gradients, because pytorch cumulates them by default.
`loss.backward` -> computes gradients via backpropagation 
`optimizser.step()` uses the gradient values, to tweak the weights to lower the loss. 

> Mental model:
> > forward pass   → what happened?
> > backward pass  → why was it wrong?
> > optimizer.step → how do we fix it?

---

## First run:

```bash

----First gen---------
At step 0, after a new line the 500 chars the model generates are:
Step 0: train loss 4.7305, val loss 4.7241

CJdgknLI!UEHwcObxfNCha!qKt-ocPTFjyXrF
W:..ZKAL.AHA.P!HAXNw,,$zCJ-!or'yxabLWGfkKowCXNe:g;gXEG'uVJMJ$
&AkfNfq-GXlay!B?!
SP JsBo.d,jIgEQzkq$YCZTOiqErphq?$zrzGJl3'IoiKIFuJuw
CM
&C-3
.yff;DRj:Td,&uDK$Wj;Y -w?XXXEG:iPDtR'd,t
-EHA3fxRObotE-wGRJiPmG'wyaIsr&NCj;IgLIas:C ?OgYQcM,jNCO.CXXgwzLlaPm$VlgY.rbXSP fyN.N&;DnQrKHX!BnyFLv$?sBtNC&x,RSAXv!
SPZBj q-rT?nQ!
oUzcZiq-UNX&vPwl:taoCJ'ZFZ-tZR?3IocmLGSVWN.s$-TAT:3!
Cj
GoV;V&QzrAYFGN&UZRR
UERjxjHH-otevlgXJMfEHRkW3f&Ch
ZgHALQeEnLv'BYwNIIN:CjXvVRmw!qAI!bsulWkXBiq
------ENTROPY---------
Shannon Entropy is : 5.904792941594465
----------------------

```

## After 3000 loop

```bash

-------------
After step 3000, after a new line the 500 chars the model generates are:
Step 2999: train loss 2.4646, val loss 2.4953

KIS:
HMa E:
Twawe highemearefid I s d
Moousthachowo.
IIthubyou! w tit far ainong m ID u
Ile I h m ick kst.
USCle Ju my hourevourkee p ce?

Jut trou t thofabout:
AUSTheas LLE:
See asey put wode at ille fot l INI the miloucisas ndor:
The
Ape's, pe w chthay arvande; vos ous panue am inghnd sanghfont wag gind t-llse?
HARUKilly'd hat hn jof CARWhe'sithas che p fe: wenghandsam sy fthenlme be,
Tut ordesssithe athethink.
cepulevor min.
Holithe, thinoyory hicocong y,
D ICI thed
SHMERD:
PEORYee
Justron t
-------------
------ENTROPY---------
Shannon Entropy is : 4.894372904441695
----------------------

```
### Observation about entropy and how to interpret

> 6 bits entropy meains: uniform choice among 65 possible characters contains about 6 bits of uncertainty

Character level Shannon entropy dropped from about 5.90 to 4.89 bits/char.
The model is ~ twice as certain about the next character than it was at initialization.
Since the theoretical maximum for 65 uniformly is ~ 6.02 bits, the step-0 model is close to random.
At step 3000, the lower entropy shows that the generated text is less random and more structured.
This does not mean fewer characters are possible, but that the character distribution is more concentrated and predictable.

> Cross-entropy does not compare the generated text to the real text directly.
It measures how much probability the model assigns to the true next token during training.
Lower cross-entropy means better next-token prediction.


> I can clearly see here that the last generation has a structure. It stil is random, but I can identify that the *little cpu* - MacBook M1 is tryting. 

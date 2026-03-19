`Thu Mar 19 09:01:11 WET 2026`
# FeedWordard

Following [video](https://youtu.be/kCc8FmEb1nY?t=5069)

```python

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

	...
class BigramLanguageModel(nn.Module):
	...
	def __init__(self):
	...
        self.ffwd = FeedForward(n_embd)
    ...
    def forward(self, idx, targets=None):
    ...
        x = tok_emb + pos_emb  # B,T,C
        x = self.sa_heads(x)
        x = self.ffwd(x)
	...
```

# Multi Layer Perceptron

> Karpathy casually says *it is just a simple Multi Layer Positron*

> He actually said **Perceptron** 

[Perceptron](https://en.wikipedia.org/wiki/Perceptron) : is an algorithm for [supervised learning](https://en.wikipedia.org/wiki/Supervised_classification "Supervised classification") of [binary classifiers](https://en.wikipedia.org/wiki/Binary_classification "Binary classification"). A binary classifier is a function that can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.

`self.net = nn.Sequential(nn.Linear(n_embd, n_embd),nn.ReLU(),)`

> For reference the ReLU notation is `(x)^+ = \max(0, x)
> Which as nothing to do with *power of* 
> And is *simply* max(0, x)

We are projecting the n_embd in the same dimension.
I would like to say that in this peculiar case we are simply *plotting*. But that would be very wrong.
> it is actually very very wrong, keep mental model of 
> **another learned linear projection in a feature space**

But, from this *plot*, applying ReLU breaks the linearity of the dimensions.
> Better said as 
> **ReLU is applied element wise and introduces nonlinearity, so the block can compute more than a single linear transformation of the input.**

**Stacking only linear layers still gives a composite function that is itself linear.
ReLU breaks that collapse by introducing nonlinearity.**

At this stage, multiple Linear projection were applied in q, k, v, that each head passed.

Each projection here was linear. 

**ReLU will clip all negative values.**

> I think here it is *hard* to see because at the moment the projection was either compressed or similar.
> But, i guess, projecting the ReLU or Q, K, V on *bigger* dimension, giving granularity, might also creates *noise*. 

> Turns out *noise* might be incorrect wording.

---

`x = tok_emb + pos_emb`
`x = self.sa_heads(x)`
`x = self.ffwd(x)`

1. First `x` is an embedding tensor of shape(B, T, C).
	Its informations are:
		- self identy
		- self position
	It has not yet mixed information from other token, *i.e.* the context in within it is, except its position.

2. Second `x` was tweak by the *self attentions heads* a.k.a the *communication channel*. 
   The embedding is now enriched with contextual awareness.
		- identity within the context 
		- which other token positions are relevant to it in this context
	It is now contextualized.

3. The third `x` is ,according to Karpathy, here to *help* the model, giving it *time* to reason before projecting into the logits <-> vocabulary.

**This is per-token computation, not token-to-token communication.**

The **MLP** does not mix **across token position**. 
And it only takes as parameters n_embed.

The feed-forward MLP does not add new context from other tokens.
Instead, it gives each token position extra nonlinear compute capacity.

**Because of the nonlinear activation, the model is not restricted to a single linear mapping from contextualized embeddings to the next representation. It can learn more complex feature transformations before the final logits projection.**

> **When we try to predict something, a simple linear trend is often the easiest baseline. But many real patterns are more complex, so we need nonlinear models to capture them. In neural nets, ReLU helps provide that extra expressive power.**



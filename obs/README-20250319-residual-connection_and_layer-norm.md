
Following [video](https://youtu.be/kCc8FmEb1nY?t=5205)

```python
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

`head_size` follows the logic discovered before:
`head_size = n_embd // n_head`

`self.ln1 = nn.LayerNorm(n_embd)`
`self.ln2 = nn.LayerNorm(n_embd)`

LayerNorm is described as complicated, but a first intuition is:
it normalizes each token's feature vector so activations stay in a within stable range.

This helps training because later computations receive inputs with a more controlled magnitude.

`x = x + self.sa(self.ln1(x))`
`x = x + self.ffwd(self.ln2(x))`

the `x + ...` part is the **residual connection**.

Each sublayer does not need to completely rebuild the representation from scratch.

> Black magic fuckery
> Somehow the model learns that sa(self.ln1(x)) does not need to carry as much signal from `x` because it is now always available........
 
Instead, it learns an update or correction to the existing representation.

So:
- LayerNorm stabilizes activations before each sublayer
- residual connections preserve an identity path and make deep training easier
  
  
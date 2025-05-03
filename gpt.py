import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# hyperparameters
seed = 1337

max_iters = 5000
eval_interval = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# ------------

torch.manual_seed(seed)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
n_vocab = len(chars)

# create a mapping from characters to integers
char2idx = { ch: i for i, ch in enumerate(chars) }
idx2char = { i: ch for i, ch in enumerate(chars) }

encode = lambda string: [char2idx[char] for char in string]
decode = lambda tensor: "".join([idx2char[idx] for idx in tensor])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

@dataclass
class GPTConfig:
    n_vocab: int
    n_embed: int = 64
    n_layer: int = 4
    n_head: int = 4
    head_size: int = n_embed // n_head
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 32 # how many independent sequences will we process in parallel?
    block_size: int = 8 # what is the maximum context length for predictions?
    
    def __post_init__(self):
        self.head_size = self.n_embed // self.n_head

config = GPTConfig(
    n_vocab=n_vocab,
    batch_size=64,
    block_size=256,
    n_embed=384,
    n_head=6,
    n_layer=6,
    lr=3e-4
)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

"""
    NOTE: 
    
    In attention calculation, the transformer head projects the input embeddings into Q, K and V
    The dimension of these are defined by "head_size"
    suppose we have 64 channel embeddings, in single head attention the head_size would be 64 / 1 = 64 as well
    
    the paper suggests that instead of having a single attention head, split this 
    
    suppose we have 8 heads and 64 embeddings, then for each head head_size would be 64 / 8 = 8
    so 8 heads perform attention in parallel using Q, K and V projected to 8 dimensional space
    
    Then the results from all these heads are concatenated, sometimes after concatenation the embedding dimension
    of the result need not match, to avoid this we project it using a final linear layer back to original embedding dim
    
    The advantage is that each head can focus on different aspects of the input data
    effectively allowing the model to attend to different representation subspaces simultaneously
"""

class Head(nn.Module):
    """
        One head of self-attention
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)        
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        
        k: torch.Tensor = self.key(x) # (B, T, head_size)
        q: torch.Tensor = self.query(x) # (B, T, head_size)
        
        # computer attention scores, affinities or weights
        
        wei: torch.Tensor = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, head_size) @ (B, head_size, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # casual self attention masking
        wei = F.softmax(wei, dim=-1) # normalize
        # randomly drop some of the affinities to prevent overfitting when it comes to large context windows
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        # can think of this as "collecting" the information from the past tokens based on the Value vector weighted by their affinities
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) ---> (B, T, head_size)
        return out
        
    
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        # projec the rejoined representation to compatible size n_embed
        self.proj = nn.Linear(config.n_head * config.head_size, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # concatenate along the channel/embed dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # (B, T, n_embed)
        return out

class FeedForward(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Singular transformer block: communication followed by computation """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # split a single communication channel into smaller ones
        head_size = config.n_embed // config.n_head
        # communication process
        self.sa = MultiHeadAttention(config)
        # computation on token wise level
        self.ffwd = FeedForward(config)
        # layer norm happens along the feature/channel/embed dimension
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        
    def forward(self, x):
        # residual connections
        # its better to add dropout before it joins the residual pathway. ie dropout before doing x + dropout<<compute>>
        x = x + self.sa(self.ln1(x)) # (B, T, n_embed)
        x = x + self.ffwd(self.ln2(x)) # (B, T, n_embed)
        
        return x
    
    


# super simple bigram model
class GPTLanguageModel(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.n_vocab, config.n_embed)
        # we need to incorporate the position of token in a block/sequence
        # transformers by itself lacks the ability to do this, so we use a position encoding
        # each position in a block has a unique position of n_embed size
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)


        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)],
        )
        
        # last layer norm after all transformer blocks
        self.ln_f = nn.LayerNorm(config.n_embed)
        # to convert token embeddings to logits, we use a linear layer
        self.lm_head = nn.Linear(config.n_embed, config.n_vocab)

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx) # (B, T, n_embed)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        
        # now x holds the token identity as well as the position
        x = token_embeddings + position_embeddings
        
        # pass through attention head
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, n_vocab)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens (context length)
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
            
        return idx

if __name__ == "__main__":
    model = GPTLanguageModel(config)
    model = model.to(device)

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    print("using config", config)
    print(model)
    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
import re

import torch
import tiktoken

BATCH_SIZE = 32 
BLOCK_SIZE = 256 
N_EMBD = 384
N_HEADS = 6
N_LAYERS = 8
DROPOUT = 0.05
WEIGHT_DECAY = 0.05

MAX_TRAIN_ITER = 10000
LR=4e-3
EVAL_ITERS = 1000


torch.manual_seed(2718)

def get_batch(split: str, train_data, val_data):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])

    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        #compute the attention scores, "affinites"
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) x (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #decoder block, "future doesn't communicate with the past"
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)

        return wei @ v
    
class MultipleHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """a super simple linear layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block: communication followed by computation"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultipleHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

def config_from_state_dict(state_dict):
    """read n_heads/n_layers back out of a checkpoint, they differ between versions"""
    n_heads = len({int(m.group(1)) for k in state_dict
                   if (m := re.match(r'blocks\.0\.sa\.heads\.(\d+)\.', k))})
    n_layers = len({int(k.split('.')[1]) for k in state_dict if k.startswith('blocks.')})
    return n_heads, n_layers


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        self.blocks = nn.Sequential(*[TransformerBlock(N_EMBD, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (batch, time, channels), (batch, block_size, channels)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop the contex
            idx_cond = idx[:, -BLOCK_SIZE:]
            #get the prediction, run forward pass
            logits, loss = self(idx_cond)
            #focus only on the last step
            logits = logits[:, -1, :]
            #apply the softmax to get the probabilities
            probs = F.softmax(logits, dim=1)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            #append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx 

if __name__ == '__main__':
    
    encoding = tiktoken.get_encoding("gpt2") #openai tokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    print('-'*20)

    with open('data/train_raw.txt', 'r') as f: 
        text = f.read().strip()

    data = torch.tensor(encoding.encode(text))

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    model = BigramLanguageModel(encoding.n_vocab).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay= WEIGHT_DECAY
    )

    for iter in range(MAX_TRAIN_ITER): 

        if iter % EVAL_ITERS == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        #we get a batch of training data
        xb, yb = get_batch('train', train_data, val_data)
        #move to gpu
        xb, yb = xb.to(device), yb.to(device)

        #evaluate it 
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        #update the weights
        optimizer.step()

    print('-'*20)
    model_cpu = model.to("cpu")
    context = torch.zeros((1, 1), dtype=torch.long)
    print(encoding.decode(model_cpu.generate(context, max_new_tokens=500)[0].tolist()))


    torch.save(model.state_dict(), 'models/model.pt') #save the model for further use
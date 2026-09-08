import torch
import tiktoken

BATCH_SIZE = 16 
BLOCK_SIZE = 8 
N_EMBD = 32

MAX_TRAIN_ITER = 5000
LR=1e-3
EVAL_ITERS = 500

encoding = tiktoken.get_encoding("cl100k_base") #openai tokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")
print('-'*20)

with open('data/train.txt', 'r') as f: 
    text = f.read().strip()

data = torch.tensor(encoding.encode(text))

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(2718)

def get_batch(split: str):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
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

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        #compute the attention scores, "affinites"
        wei = q @ k.transpose(-2, -1) * C **-0.5# (B, T, 16) x (B, 16, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #decoder block, "future doesn't communicate with the past"
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)

        return wei @ v
    

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.attention_head = Head(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (batch, time, channels), (batch, block_size, channels)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.attention_head(x)
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


model = BigramLanguageModel(encoding.n_vocab).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay= 0.01
)

for iter in range(MAX_TRAIN_ITER): 

    if iter % EVAL_ITERS == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #we get a batch of training data
    xb, yb = get_batch('train')
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
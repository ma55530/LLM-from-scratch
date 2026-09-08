import torch
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
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

BATCH_SIZE = 16 
BLOCK_SIZE = 8 
N_EMBDS = 128
MAX_TRAIN_ITER = 3000
EVAL_ITERS = 20

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

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBDS)
        self.lm_head = nn.Linear(N_EMBDS, vocab_size)
    
    def forward(self, idx, targets=None):

        #idx and targets are both (B,T) tensor of integers
        x = self.token_embedding_table(idx) # (batch, time, channels), (batch, block_size, channels)
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
            #get the prediction, run forward pass
            logits, loss = self(idx)
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
    lr=2e-3,
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
import sys
import webbrowser
from pathlib import Path

import torch
import tiktoken
import circuitsvis as cv
from nnsight import NNsight

from bigram import BigramLanguageModel, config_from_state_dict

PROMPT = input("Type in the prompt: ")
LAYER = int(input("Layer: "))
OUT = Path('attention.html')

encoding = tiktoken.get_encoding("gpt2") #openai tokenizer
state_dict = torch.load('models/scratch_v2.pt', map_location='cpu')
n_heads, n_layers = config_from_state_dict(state_dict)
model = BigramLanguageModel(encoding.n_vocab, n_heads, n_layers)
model.load_state_dict(state_dict)
model.eval() #dropout off, so the saved dropout output is the raw attention

nnsight_model = NNsight(model)

ids = encoding.encode(PROMPT)
tokens = torch.tensor(ids, dtype=torch.long)[None, :]
str_tokens = [encoding.decode([i]) for i in ids] #circuitsvis wants strings, not ids

attn = []
with nnsight_model.trace(tokens):
    for h in range(n_heads):
        attn.append(nnsight_model.blocks[LAYER].sa.heads[h].dropout.output.save())

#each head is (B, T, T), we want (n_heads, dest, src)
patterns = torch.stack([a[0] for a in attn])

html = cv.attention.attention_patterns(attention=patterns, tokens=str_tokens)
OUT.write_text(str(html))
print(f"layer {LAYER}, {n_heads} heads, {len(str_tokens)} tokens -> {OUT.resolve()}")
webbrowser.open(OUT.resolve().as_uri())

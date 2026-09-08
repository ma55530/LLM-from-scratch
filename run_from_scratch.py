import torch
from bigram import BigramLanguageModel
import tiktoken
import sys
MAX_NEW_TOKENS = 30

encoding = tiktoken.get_encoding("gpt2") #openai tokenizer

model = BigramLanguageModel(encoding.n_vocab)
model.load_state_dict(torch.load('models/scratch_v2.pt', map_location='cpu'))

model.eval()   # disables dropout for inference

# for p in model.parameters():
#     print(p.numel())
print(f"From scratch model with {sum(p.numel() for p in model.parameters())/1e6} M parameters\n")

while True:
    user_input = input('> ')

    if not user_input.strip():
        continue

    if user_input == 'exit' or user_input == 'quit':
        sys.exit() 

    tokens = encoding.encode(user_input)
    idx = torch.tensor(tokens, dtype=torch.long)[None, :]

    output = model.generate(idx, MAX_NEW_TOKENS)[0].tolist()[len(tokens):]

    messages = [m for m in encoding.decode(output).split('\n') if m.strip()]

    print('\n'.join(messages[:2]))
    print()
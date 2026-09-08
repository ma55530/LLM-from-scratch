from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import sys

# to train, run:
#       mlx_lm.lora --model mlx-community/Ministral-8B-Instruct-2410-4bit --data data/QA --train --config lora_config.yaml 
model, tokenizer = load("mlx-community/Ministral-8B-Instruct-2410-4bit", adapter_path="adapters")


sampler = make_sampler(temp=0.6, top_p=0.7)

while True:
    user_input = input("> ")

    if not user_input:
        continue

    if user_input == 'exit' or user_input == 'quit':
        sys.exit() 
    
    messages = [{"role": "user", "content" : user_input}]
    tokenized_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    out = generate(model, tokenizer, prompt=tokenized_text, max_tokens=500, sampler=sampler)

    print(out)
    print()
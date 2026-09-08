# LanguageModels

Two language models trained on my own WhatsApp and Instagram message history:

1. **From scratch** — a ~50M parameter decoder-only transformer (GPT-style), written in plain PyTorch and trained on my raw message text.
2. **Fine-tuned** — [Ministral-8B-Instruct](https://huggingface.co/mlx-community/Ministral-8B-Instruct-2410-4bit) fine-tuned with LoRA on question/answer pairs extracted from my chats, using [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon.

The training data itself is personal and is not included in this repo — only the code and the trained weights.

## Project structure

```
bigram.py              from-scratch transformer: model definition + training loop
run_from_scratch.py    interactive REPL for the from-scratch model
run_fine_tuned.py      interactive REPL for the LoRA fine-tuned Ministral-8B
lora_config.yaml       mlx-lm LoRA training configuration
attention.py           attention visualization experiment (see attention.html)
bigram_archive/        earlier versions of the from-scratch model (v1–v3)
adapters/              final LoRA adapter weights + config
data/                  data extraction/processing scripts (raw data not included)
```

## From-scratch model

Decoder-only transformer, roughly following Karpathy's *"Let's build GPT"*, but using the GPT-2 BPE tokenizer (`tiktoken`) instead of a character-level vocabulary:

- 6 transformer blocks, 4 heads, 384-dim embeddings, 256-token context
- GPT-2 tokenizer (50257 vocab)
- trained on ~2.5MB of my own message text

Train (expects `data/train_raw.txt`):

```
python bigram.py
```

Chat with it:

```
python run_from_scratch.py
```

The trained checkpoint `scratch_v2.pt` is too large for git (~200MB) — download it from this repo's **Releases** page and place it in `models/`.

## Fine-tuned model

LoRA fine-tune of Ministral-8B-Instruct (4-bit) on Q/A pairs built from my WhatsApp conversations (messages to me = question, my replies = answer).

Train:

```
mlx_lm.lora --model mlx-community/Ministral-8B-Instruct-2410-4bit --data data/QA --train --config lora_config.yaml
```

Chat with it (the base model downloads from Hugging Face automatically; the adapter is included in `adapters/`):

```
python run_fine_tuned.py
```

## Data pipeline

Scripts in `data/` (the data they operate on is not included):

- `extract_wapp_text.py` — parse WhatsApp chat exports into raw text and Q/A pairs
- `extract_ig_text.py` — parse Instagram message/comment exports into raw text
- `raw_data.py` — combine sources into `train_raw.txt` for from-scratch training
- `QA.py` — shuffle and split Q/A pairs into train/valid/test jsonl
- `analyzeWapp.py` — sanity checks and stats on the Q/A dataset

## Setup

```
pip install -r requirements.txt
```

`mlx-lm` requires Apple Silicon; the from-scratch model runs anywhere (uses MPS if available, otherwise CPU).

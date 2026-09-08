import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import torch
import tiktoken
from nnsight import NNsight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import bigram.py from the project root
from bigram import BigramLanguageModel, config_from_state_dict, BLOCK_SIZE

PROMPTS = [
    "volim te jako puno",
    "HAHAHAHAHA",
    "kako si ti chill lik",
    "U ponedjeljak dolazim u zagreb",
    "imam tjedan dana za to napraviti",
    "Ocemo u nedjelju u maksimir ici se setati",
    "Na faksu sam.",
    "Laku noc mico",
    "ne mogu danas imam kolokvij iz matematike pa moram ucit cijeli dan",
    "javi mi kad stignes doma da znam da si dobro dosao",
    "di si\nsta ima\nnis samo doma\njel idemo veceras van\nne znam jos vidjet cu",
    "jesi vidio ono sto sam ti poslao jucer navecer\nnisam stigao\npogledaj kad budes imao vremena",
    "sutra ujutro imam predavanje u 8 pa necu moci doci prije 11 mislim",
    "hvala ti puno na svemu stvarno mi je puno znacilo sto si bio tu",
    "ma daj ne seri\nozbiljno ti kazem\nne vjerujem ti nista vise",
    "koliko je sati\npola 4\njoj kasnim moram ic",
]

OUT = Path("assets/attention_map.png")


def load_model(model_name):
    state_dict = torch.load(model_name, map_location="cpu")
    n_heads, n_layers = config_from_state_dict(state_dict)
    model = NNsight(BigramLanguageModel(tiktoken.get_encoding("gpt2").n_vocab, n_heads, n_layers))
    model.load_state_dict(state_dict)
    model.eval()  # dropout off, so what we capture is the raw softmax output

    return model, n_heads, n_layers


def capture_attention(model, tokens, n_heads, n_layers):

    attn = []
    with model.trace(tokens):
        for layer in range(n_layers):
            for head in range(n_heads):
                attn.append(model.blocks[layer].sa.heads[head].dropout.output.save())

    attn = torch.stack([x[0].detach() for x in attn])
    return attn.reshape(n_layers, n_heads, *attn.shape[-2:])

def head_stats(attn):
    T = attn.shape[-1]
    if T < 2:
        zero = torch.zeros(attn.shape[:-2])
        return {k: zero for k in ("distance", "sink", "prev", "self")}

    pos = torch.arange(T)
    offsets = (pos[:, None] - pos[None, :]).clamp(min=0)
    i = pos[1:]  # row 0 is trivially all self-attention, skip it

    return {
        "distance": (attn * offsets).sum(-1)[..., 1:].mean(-1),
        "sink": attn[..., 1:, 0].mean(-1),
        "prev": attn[..., i, i - 1].mean(-1),
        "self": attn[..., i, i].mean(-1),
    }


def pool_stats(per_prompt, lengths):

    weights = torch.tensor([max(T - 1, 0) for T in lengths], dtype=torch.float)
    weights = weights / weights.sum()
    return {k: sum(w * p[k] for w, p in zip(weights, per_prompt)) for k in per_prompt[0]}


def pad_to(attn, T):
    padded = torch.full((*attn.shape[:2], T, T), float("nan"))
    t = attn.shape[-1]
    padded[..., :t, :t] = attn
    return padded


def plot_grid(mean, n_heads, n_layers, n_prompts, out_path):
    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(2.2 * n_heads, 2.2 * n_layers), squeeze=False)

    cmap = matplotlib.cm.viridis.copy()

    cmap.set_bad("lightgrey")

    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l][h]
            ax.imshow(mean[l, h].numpy(), cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"L{l} H{h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"mean attention over {n_prompts} prompts, padded to the longest\n"
                 "row = query position, column = attended position", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":

    torch.set_grad_enabled(False)
    encoding = tiktoken.get_encoding("gpt2")
    model, n_heads, n_layers = load_model("models/scratch_v2.pt")

    A = []

    for prompt in PROMPTS:

        ids = encoding.encode(prompt)
        if len(ids) > BLOCK_SIZE:
            ids = ids[:BLOCK_SIZE]

        tokens = torch.tensor(ids, dtype=torch.long)[None, :]
        attention = capture_attention(model, tokens, n_heads, n_layers)
        A.append(attention)
    
    Tmax = max(a.shape[-1] for a in A)
    mean = torch.stack([pad_to(a, Tmax) for a in A]).nanmean(0)

    #the rest is vibe-coded
    plot_grid(mean, n_heads, n_layers, len(A), OUT)
    print(f"\n{n_layers} layers x {n_heads} heads, up to {Tmax} tokens -> {OUT.resolve()}")

    # stats are pooled from per-prompt values
    stats = pool_stats([head_stats(a) for a in A], [a.shape[-1] for a in A])

    print()
    print(f"{'head':>8}{'distance':>10}{'sink':>8}{'prev':>8}{'self':>8}")
    for l in range(n_layers):
        for h in range(n_heads):
            print(f"{f'L{l}H{h}':>8}{stats['distance'][l, h]:>10.2f}{stats['sink'][l, h]:>8.2f}"
                f"{stats['prev'][l, h]:>8.2f}{stats['self'][l, h]:>8.2f}")

    print()
    print("per-layer averages")
    print(f"{'layer':>8}{'distance':>10}{'sink':>8}{'prev':>8}")
    for l in range(n_layers):
        print(f"{l:>8}{stats['distance'][l].mean():>10.2f}"
            f"{stats['sink'][l].mean():>8.2f}{stats['prev'][l].mean():>8.2f}")

    
    print()
    l, h = divmod(stats["sink"].argmax().item(), n_heads)
    print(f"strongest first-token head: L{l}H{h} "
        f"({stats['sink'][l, h]:.0%} of attention on the preceding token)")
    print()
    l, h = divmod(stats["self"].argmax().item(), n_heads)
    print(f"strongest self-token head: L{l}H{h} "
        f"({stats['self'][l, h]:.0%} of attention on the preceding token)")
    print()
    l, h = divmod(stats["prev"].argmax().item(), n_heads)
    print(f"strongest previous-token head: L{l}H{h} "
        f"({stats['prev'][l, h]:.0%} of attention on the preceding token)")
    

import json, random

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

data = read_jsonl("data/QA/train.jsonl")
random.seed(0)
for ex in random.sample(data, 40):
    u = ex["messages"][0]["content"]
    a = ex["messages"][1]["content"]
    print(f"Q: {u!r}\nA: {a!r}\n{'-'*40}")


from collections import Counter

lens_q = [len(e["messages"][0]["content"]) for e in data]
lens_a = [len(e["messages"][1]["content"]) for e in data]

def pctile(xs, p): 
    return sorted(xs)[int(len(xs)*p)]

print("answer len  p10/p50/p90/p99:",
      *(pctile(lens_a, p) for p in (.1,.5,.9,.99)))

# how much is trivial repetition?
answers = Counter(e["messages"][1]["content"] for e in data)
questions = Counter(e["messages"][0]["content"] for e in data)

print("top repeated questions:", questions.most_common(15))
print("top repeated answers:", answers.most_common(15))

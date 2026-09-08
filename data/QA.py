import json
import random

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == '__main__':

    with open('data/QA/raw/wapp.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

        random.shuffle(data)

        n = len(data)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train = data[:train_end]
        val   = data[train_end:val_end]
        test  = data[val_end:]

        write_jsonl("data/QA/train.jsonl", train)
        write_jsonl("data/QA/valid.jsonl",   val)
        write_jsonl("data/QA/test.jsonl",  test)

        print(f"total={n}  train={len(train)}  val={len(val)}  test={len(test)}")

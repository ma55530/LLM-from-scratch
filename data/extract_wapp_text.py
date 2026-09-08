import regex as re
from pathlib import Path
import json
from collections import Counter

MY_NAME = "Matija Akrap"
def ExtractWappDataRaw():
    newlines = []

    dir = Path("data/whatsapp_data")
    for file in dir.iterdir():
        if file.is_file():
            print(f"Processing {file}...")
            with open(file, 'r') as f:
                lines = f.read().split('\n')
                
                for text in lines:
                    
                    ptrn = f"] {MY_NAME}:"
                    if ptrn in text:
                        if(' omitted' in text
                            or 'Video call.' in text
                            or 'No answer.' in text
                            or '‎' in text):
                            continue
                        
                        text = re.sub(rf"\[.*?\]\s*{re.escape(MY_NAME)}:\s*", "", text)
                        text = re.sub(r'https?://\S+|www\.\S+', '', text)
                        
                        try:
                            text = text.encode("latin1").decode("utf-8")
                        except:
                            try:
                                re.sub(r"?:\\u[0-9a-fA-F]{4}+", "", text)
                            except:
                                pass
                            pass
                        newlines.append(text)



    f = open('data/raw/wapp.txt', 'w')
    for line in newlines:
        put = line + "\n"
        f.write(put)


def ExtractWappQA():
    
    qa = []

    message_start = re.compile(
        r'^\[(\d{2}\.\d{2}\.\d{4}\.), (\d{2}:\d{2}:\d{2})\] (.*?): (.*?)'
        r'(?=^\[\d{2}\.\d{2}\.\d{4}\.,\s\d{2}:\d{2}:\d{2}\]|\Z)',
        re.MULTILINE | re.DOTALL,
    )
    SKIP = {"<Media omitted>", "This message was deleted",
            "This message was edited", "null", "<This message was edited>",}
    
    def pair(q, a):
        return {"messages": [
            {"role": "user", "content": "\n".join(q)},
            {"role": "assistant", "content": "\n".join(a)},
        ]}
    

    dir = Path("data/whatsapp_data")
    for file in dir.iterdir():
        if file.is_file():
            print(f"Processing {file}...")

            messages = []

            with open(file, 'r') as f:
                text = f.read()

                for message in message_start.finditer(text):
                    body = message.group(4).strip()
                    if '‎' in body:
                        continue
                    
                    body = re.sub(r'https?://\S+|www\.\S+', '', body).strip()
                    if len(body) < 8 or body in SKIP:
                        continue

                    if body:
                        messages.append({"sender": message.group(3), "text": body})
              
            q_buffer = []
            a_buffer = []
            a_was_last = False

            for msg in messages:
                if msg["sender"] != MY_NAME:
                    if a_was_last:
                        if q_buffer and a_buffer:
                            example = pair(q_buffer, a_buffer)
                            qa.append(example)
                            
                            q_buffer = []
                            a_buffer = []

                    q_buffer.append(msg['text'])
                    a_was_last = False
                else:
                    a_buffer.append(msg['text'])
                    a_was_last = True

            if q_buffer and a_buffer:
                example = pair(q_buffer, a_buffer)
                qa.append(example)


    with open('data/QA/raw/wapp.jsonl', 'w', encoding='utf-8') as f:
        for item in qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    ExtractWappQA()
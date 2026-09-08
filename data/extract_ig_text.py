import regex as re
from pathlib import Path
import json
import os


def ExtractInstagramDataRaw():
    newlines = []
    # inbox parsing 
    inbox_dir = Path("data/instagram_data/inbox")
    for inbox in inbox_dir.iterdir():
        if inbox.is_dir(): #person username some code 
            for file in inbox.iterdir(): 
                if file.is_file():
                    print(f"Processing {file}...")

                    with open(file, 'r') as jsonfile:
                        data = json.load(jsonfile)

                        messages = data['messages']

                        for message in messages:
                            if message["sender_name"] == "Matija Akrap":
                                try:
                                    content = message["content"]

                                    if(
                                        'You sent an attachment.' in content
                                        or 'You started a' in content
                                        or 'Audio call ended' in content
                                        or 'Video chat ended' in content
                                    ):
                                        continue

                                    content = re.sub(r'https?://\S+|www\.\S+', '', content)

                                    content = content.encode("latin1").decode("utf-8")

                                    newlines.append(content)
                                except:
                                    pass


    comments_dir = Path('data/instagram_data/comments')

    for file in comments_dir.iterdir():
        if file.is_file():
            if 'post_comments' in os.path.basename(file):
                with open(file, 'r') as jsonfile:
                        data = json.load(jsonfile)

                        for comment in data:
                            try:
                                text = comment["string_map_data"]["Comment"]["value"]
                                text = text.encode("latin1").decode("utf-8")
                                text = re.sub(r"@\w+", "", text)
                                newlines.append(text)
                            except:
                                pass

            if 'reels_comments' in os.path.basename(file):
                with open(file, 'r') as jsonfile:
                        data = json.load(jsonfile)

                        reels_comments = data["comments_reels_comments"]
                        
                        for comment in reels_comments:
                            text = comment["string_map_data"]["Comment"]["value"]
                            text = text.encode("latin1").decode("utf-8")
                            text = re.sub(r"@\w+", "", text)
                            newlines.append(text)

    f = open('data/raw/insta.txt', 'w')
    for line in newlines:
        put = line + "\n"
        f.write(put)


if __name__ == '__main__':
    ExtractInstagramDataRaw()
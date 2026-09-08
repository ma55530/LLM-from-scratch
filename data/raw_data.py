

from extract_ig_text import ExtractInstagramDataRaw
from extract_wapp_text import ExtractWappDataRaw


if __name__ == '__main__':

    ExtractWappDataRaw()
    ExtractInstagramDataRaw()

    filenames = ['data/raw/insta.txt', 'data/raw/wapp.txt']
    with open('data/train_raw.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
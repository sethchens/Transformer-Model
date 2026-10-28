import os
import numpy as np
from gpt.tok import get_tokenizer

CHUNK_CHARS = 64_000_000   # ~64 MB of text per read
SEP = "<|endoftext|>"


def write_stories(stories, enc, out):
    """Tokenize a list of stories, add EOT after each, write as uint16. Returns token count."""
    if not stories:
        return 0
    ids = []
    for toks in enc.encode_ordinary_batch(stories):
        ids.extend(toks)
        ids.append(enc.eot_token)
    arr = np.array(ids, dtype=np.uint16)
    assert arr.max() < 2**16
    arr.tofile(out)
    return len(ids)


def tokenize_file(src, dst):
    enc = get_tokenizer()
    leftover = ""
    total = 0

    with open(src, "r", encoding="utf-8") as f, open(dst, "wb") as out:
        while True:
            piece = f.read(CHUNK_CHARS)
            if not piece:
                break
            stories = (leftover + piece).split(SEP)
            leftover = stories.pop()                      # may be cut mid-story
            stories = [s.strip() for s in stories if s.strip()]
            total += write_stories(stories, enc, out)
            print(f"{dst}: {total:,} tokens")

        # last story (nothing after it to trigger the split)
        if leftover.strip():
            total += write_stories([leftover.strip()], enc, out)

    print(f"done {dst}: {total:,} tokens, {os.path.getsize(dst)/1e6:.1f} MB")
    return total


def check(path, n=100):
    enc = tiktoken.get_encoding("gpt2")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    print(f"{path}: {len(data):,} tokens")
    print(enc.decode(data[:n].tolist()))


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    tokenize_file("data/TinyStoriesV2-GPT4-valid.txt", "data/test.bin")
    check("data/test.bin")
    tokenize_file("data/TinyStoriesV2-GPT4-train.txt", "data/train.bin")
    check("data/train.bin")
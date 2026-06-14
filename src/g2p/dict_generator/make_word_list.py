import os, re, argparse
from collections import Counter

def extract_words_from_text(text):
    return re.findall(r"[A-Za-z']+", text)

def build_word_list(corrected_dir, out_path="words.txt"):
    counts = Counter()
    for fn in os.listdir(corrected_dir):
        if not fn.lower().endswith(".txt"):
            continue
        with open(os.path.join(corrected_dir, fn), "r", encoding="utf8") as fh:
            txt = fh.read().strip()
        words = [w.lower() for w in extract_words_from_text(txt)]
        counts.update(words)
    with open(out_path, "w", encoding="utf8") as out:
        for w, c in counts.most_common():
            out.write(w + "\n")
    print(f"Wrote {len(counts)} unique words to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corrected_dir", required=True)
    p.add_argument("--out", default="words.txt")
    args = p.parse_args()
    build_word_list(args.corrected_dir, args.out)

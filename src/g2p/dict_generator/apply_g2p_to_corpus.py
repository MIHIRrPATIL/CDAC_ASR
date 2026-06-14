# apply_g2p_to_corpus.py
import os, re, argparse

def load_dict(dict_path):
    # expects lines: word [tab or space] phoneme1 phoneme2 ...
    mapping = {}
    with open(dict_path, "r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # sometimes outputs: WORD <tab> PH1 PH2 ...  OR WORD  PH...
            word = parts[0].lower()
            phones = parts[1:]
            # if there's a score column at the end (from --export_scores), remove numeric
            if phones and phones[-1].replace('.', '', 1).isdigit():
                phones = phones[:-1]
            mapping[word] = phones
    return mapping

def tokenize(text):
    return re.findall(r"[A-Za-z']+", text)

def apply_g2p_to_corpus(corrected_dir, dict_path, out_dir="phonemes", unk_token="<UNK>"):
    os.makedirs(out_dir, exist_ok=True)
    mapping = load_dict(dict_path)
    missing = set()
    files = [f for f in os.listdir(corrected_dir) if f.lower().endswith(".txt")]
    for fn in files:
        sid = os.path.splitext(fn)[0]
        with open(os.path.join(corrected_dir, fn), "r", encoding="utf8") as fh:
            txt = fh.read().strip().lower()
        words = tokenize(txt)
        phs = []
        for w in words:
            if w in mapping:
                phs.extend(mapping[w])
            else:
                missing.add(w)
                phs.append(unk_token)
        out_path = os.path.join(out_dir, sid + ".phn")
        with open(out_path, "w", encoding="utf8") as oh:
            oh.write(" ".join(phs))
    print(f"Wrote {len(files)} .phn files to {out_dir}")
    if missing:
        print("Missing words (not in dictionary) — consider running mfa g2p on them or adding manually:")
        print(", ".join(sorted(list(missing))[:100]))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corrected_dir", required=True)
    p.add_argument("--dict", required=True, help="output.dict from mfa g2p")
    p.add_argument("--out_dir", default="phonemes")
    args = p.parse_args()
    apply_g2p_to_corpus(args.corrected_dir, args.dict, args.out_dir)

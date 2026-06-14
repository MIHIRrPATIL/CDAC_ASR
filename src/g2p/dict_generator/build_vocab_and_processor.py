# python build_vocab_and_processor.py --phn_dir phonemes --out_dir processor_dir
import os, json, argparse
from collections import Counter
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

def build_vocab(phn_dir):
    tokens = Counter()
    for fn in os.listdir(phn_dir):
        if not fn.endswith(".phn"):
            continue
        with open(os.path.join(phn_dir, fn), "r", encoding="utf8") as fh:
            toks = fh.read().strip().split()
            tokens.update(toks)
    # special tokens: pad and unk (pad used as CTC blank)
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for t in sorted(tokens):
        if t in vocab:
            continue
        vocab[t] = idx
        idx += 1
    return vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phn_dir", required=True)
    parser.add_argument("--out_dir", default="processor_dir")
    parser.add_argument("--feature_extractor_from", default="facebook/wav2vec2-base-960h")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vocab = build_vocab(args.phn_dir)
    vocab_path = os.path.join(args.out_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf8") as fh:
        json.dump(vocab, fh, indent=2)
    print("Wrote vocab.json with", len(vocab), "tokens")

    # create tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    # reuse a pretrained feature extractor (downloads from HF if needed)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.feature_extractor_from)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(args.out_dir)
    print("Saved processor to", args.out_dir)

    # save mapping for later
    phoneme2id = vocab
    with open(os.path.join(args.out_dir, "phoneme2id.json"), "w", encoding="utf8") as fh:
        json.dump(phoneme2id, fh, indent=2)
    print("Saved phoneme2id.json")

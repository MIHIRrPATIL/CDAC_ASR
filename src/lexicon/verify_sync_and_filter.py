import os
import json

def main():
    dict_path = "src/lexicon/output/ie_detailed_ipa_lexicon.dict"
    vocab_path = "models/processor_dir/vocab.json"

    if not os.path.exists(dict_path):
        print(f"Error: Dictionary not found at {dict_path}")
        return

    if not os.path.exists(vocab_path):
        print(f"Error: Vocab not found at {vocab_path}")
        return

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Loaded vocab with {len(vocab)} tokens.")

    # Parse dictionary and check tokens
    missing_tokens = set()
    total_words = 0
    longest_word = None
    longest_length = 0
    length_distribution = []

    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word = parts[0]
            phones = parts[1].split()
            total_words += 1

            length_distribution.append((word, len(phones)))
            if len(phones) > longest_length:
                longest_length = len(phones)
                longest_word = (word, phones)

            for phone in phones:
                if phone not in vocab:
                    missing_tokens.add(phone)

    print(f"\n--- Vocab Synchronization Check ---")
    print(f"Total words in dictionary: {total_words}")
    if missing_tokens:
        print(f"❌ FAILED: Found {len(missing_tokens)} missing tokens in vocab.json:")
        for t in sorted(list(missing_tokens)):
            print(f"  - {repr(t)}")
    else:
        print("✅ SUCCESS: All dictionary tokens are present in vocab.json!")

    print(f"\n--- Dynamic Range Filtering Check ---")
    print(f"Longest phoneme sequence length: {longest_length} tokens (Word: '{longest_word[0]}')")
    print(f"Phonemes: {longest_word[1]}")

    # Show top 10 longest words
    print("\nTop 10 longest words in dictionary:")
    length_distribution.sort(key=lambda x: x[1], reverse=True)
    for w, l in length_distribution[:10]:
        print(f"  - '{w}': {l} phonemes")

if __name__ == "__main__":
    main()

import argparse
from g2p_utils import G2PManager

def main():
    parser = argparse.ArgumentParser(description="Test G2P pronunciation lookup.")
    parser.add_argument("text", help="Sentence or words to convert to phonemes")
    parser.add_argument("--dict", help="Optional path to custom dictionary")
    args = parser.parse_args()

    g2p = G2PManager(dict_path=args.dict)
    
    # 1. Phonemize
    phonemes = g2p.convert_sentence(args.text)
    
    # 2. Display
    print("\n" + "="*50)
    print(f"INPUT:    {args.text}")
    print(f"PHONEMES: {' '.join(phonemes)}")
    print("="*50 + "\n")

    # 3. Validation report
    words = g2p.tokenize(args.text)
    missing = [w for w in words if w.lower() not in g2p.phoneme_dict]
    if missing:
        print(f"⚠ WARNING: The following words were not found in the dictionary: {', '.join(missing)}")
        print("These will be treated as 'Identity' phonemes during training/evaluation.")
    else:
        print("✓ All words found in dictionary.")

if __name__ == "__main__":
    main()

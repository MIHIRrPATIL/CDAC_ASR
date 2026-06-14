import os
import re
import argparse
from tqdm import tqdm
from src.g2p.g2p_utils import G2PManager
from evaluate_indian_accent import extract_transcript
import nltk

def parse_args():
    parser = argparse.ArgumentParser(description="Verify G2P dictionary coverage and fallbacks")
    parser.add_argument("--dataset_dir", default="indian-accent-dataset/audio", help="Path to Kaggle dataset splits")
    parser.add_argument("--dict_path", default="src/g2p/output_v2_detailed.dict", help="Path to dictionary")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Download required NLTK resources
    print("Checking NLTK resources...")
    for res in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']:
        nltk.download(res, quiet=True)

    
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Error: Dataset directory '{args.dataset_dir}' not found.")
        return
        
    g2p = G2PManager(dict_path=args.dict_path)
    
    # Track statistics
    total_words = 0
    dict_hits = 0
    neural_fallbacks = 0
    identity_fallbacks = 0
    
    fallback_examples = set()
    identity_examples = set()
    
    # Walk the dataset splits to extract all transcripts
    splits = ["train", "test", "dev"]
    transcripts = []
    
    for split in splits:
        split_dir = os.path.join(args.dataset_dir, "audio", split)
        if not os.path.exists(split_dir):
            split_dir = os.path.join(args.dataset_dir, split)
            if not os.path.exists(split_dir):
                continue
                
        speaker_dirs = [
            os.path.join(split_dir, d) for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ]
        
        for sd in speaker_dirs:
            t = extract_transcript(sd)
            if t:
                transcripts.append(t)
                
    print(f"📄 Found {len(transcripts)} transcripts. Tokenizing words...")
    
    for sentence in tqdm(transcripts):
        words = g2p.tokenize(sentence)
        for word in words:
            total_words += 1
            word_lower = word.lower()
            
            # 1. Dictionary check
            if word_lower in g2p.phoneme_dict:
                dict_hits += 1
            else:
                # 2. Neural Fallback check
                if g2p.neural_g2p is not None:
                    neural_fallbacks += 1
                    fallback_examples.add(word_lower)
                else:
                    identity_fallbacks += 1
                    identity_examples.add(word_lower)
                    
    # Print report
    print("\n" + "="*50)
    print("            G2P DIAGNOSTIC REPORT")
    print("="*50)
    print(f"Total Words Processed:    {total_words}")
    if total_words > 0:
        print(f"Dictionary Hits:          {dict_hits} ({dict_hits/total_words:.2%})")
        print(f"Neural (g2p-en) Fallbacks: {neural_fallbacks} ({neural_fallbacks/total_words:.2%})")
        print(f"Identity Fallbacks:       {identity_fallbacks} ({identity_fallbacks/total_words:.2%})")
    print("="*50)
    
    if fallback_examples:
        print("\n💡 Sample words that fell back to Neural G2P:")
        print(list(fallback_examples)[:30])
        
    if identity_examples:
        print("\n🚨 Sample words that had NO fallback (Identity mapping):")
        print(list(identity_examples)[:30])

if __name__ == "__main__":
    main()

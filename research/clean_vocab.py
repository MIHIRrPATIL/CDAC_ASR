import os
import sys
import json
import re
from datasets import load_dataset, Audio
import nltk

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g2p.g2p_utils import G2PManager

# Define datasets to inspect
DATASETS = {
    "NPTEL": ("skbose/indian-english-nptel-v0", "train", "transcription"),
    "EkaCare": ("ekacare/eka-medical-asr-evaluation-dataset", "test", "transcript"),
    "theothertom": ("theothertom/indian_english_audio_2", "train", "transcription"),
    "CommonVoice": ("WillHeld/india_accent_cv", "train", "sentence")
}

def main():
    print("Checking NLTK resources...")
    for res in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

    dict_path = "research/g2p/output_full.dict"
    if not os.path.exists(dict_path):
        dict_path = "g2p/output_full.dict"
        if not os.path.exists(dict_path):
            print("❌ Error: G2P Dictionary not found.")
            return

    print(f"Loading G2P Manager with dict: {dict_path}")
    g2p = G2PManager(dict_path=dict_path)
    
    # Track OOVs
    oov_words = set()
    total_words_checked = 0

    for name, (path, split, text_col) in DATASETS.items():
        print(f"\n📂 Loading and analyzing text from dataset: {name} ({path})...")
        try:
            # Load dataset in streaming mode to scan transcripts quickly
            ds = load_dataset(path, split=split, streaming=True)
            # Disable audio decoding to bypass FFmpeg/torchcodec loader issues
            if "audio" in ds.features:
                ds = ds.cast_column("audio", Audio(decode=False))
            
            # Read first 5000 transcripts per dataset for OOV audit
            count = 0
            for sample in ds:
                text = sample.get(text_col, "")
                if not text:
                    # Dynamically look for common text keys if standard key is missing
                    for key in ["text", "transcript", "transcription", "sentence"]:
                        if key in sample:
                            text = sample[key]
                            break
                
                text_str = str(text).strip()
                if not text_str:
                    continue

                words = g2p.tokenize(text_str)
                for word in words:
                    total_words_checked += 1
                    # A word is OOV if it's not in the dictionary mapping
                    if word not in g2p.phoneme_dict:
                        # Check if neural fallback can handle it
                        if g2p.neural_g2p is not None:
                            # g2p-en returns a list of phonetic tokens. If empty or returns original word, it's a failure.
                            try:
                                fallback_phn = g2p.neural_g2p(word)
                                # If neural G2P just echoes the word or is empty, it failed
                                if not fallback_phn or (len(fallback_phn) == 1 and fallback_phn[0].lower() == word):
                                    oov_words.add(word)
                            except Exception:
                                oov_words.add(word)
                        else:
                            oov_words.add(word)
                
                count += 1
                if count >= 5000:
                    break
                    
            print(f"✓ Scanned {count} samples from {name}.")
        except Exception as e:
            print(f"⚠️ Failed to scan dataset {name}: {e}")

    print("\n" + "="*50)
    print("           VOCABULARY G2P DIAGNOSTIC REPORT")
    print("="*50)
    print(f"Total words checked:  {total_words_checked}")
    print(f"Unique OOV words:     {len(oov_words)}")
    
    if oov_words:
        print("\n📝 Sample OOV words (will cause <unk> if not patched):")
        for w in sorted(list(oov_words))[:20]:
            print(f"  - {w}")
            
        # Write to a patch file
        patch_file = "research/g2p/patch_vocab.txt"
        print(f"\n💾 Writing OOV word list to {patch_file}...")
        os.makedirs(os.path.dirname(patch_file), exist_ok=True)
        with open(patch_file, "w", encoding="utf-8") as f:
            for w in sorted(list(oov_words)):
                f.write(f"{w}\n")
        print("✓ Done! You can use this list to generate IPA pronunciations and append them to output_full.dict.")
    else:
        print("\n🎉 Zero OOV words detected! The dictionary covers 100% of the vocabulary.")
    print("="*50)

if __name__ == "__main__":
    main()

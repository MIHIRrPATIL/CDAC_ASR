import os
import sys
import argparse
import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
import Levenshtein
import json

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import Wav2Vec2Processor
from src.models.phoneme_embedder import Wav2Vec2PhonemeEmbedder

def calculate_per(reference, hypothesis):
    """Memory-efficient sequence-level Levenshtein distance for PER."""
    nr = len(reference)
    nh = len(hypothesis)
    if nr == 0: return nh
    if nh == 0: return nr
    
    row = np.arange(nh + 1)
    for i in range(1, nr + 1):
        prev_row = row.copy()
        row[0] = i
        for j in range(1, nh + 1):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            row[j] = min(prev_row[j] + 1,      # deletion
                         row[j-1] + 1,          # insertion
                         prev_row[j-1] + cost) # substitution
    
    return row[nh] / nr

def main():
    parser = argparse.ArgumentParser(description="Evaluate Phoneme Embedder on local preprocessed test split")
    parser.add_argument("--dataset_dir", default="/data/local_nptel_processed", help="Path to local processed dataset")
    parser.add_argument("--model_dir", default="/data/nptel_embedder_checkpoints/early_stop_health_check", help="Path to model checkpoint")
    parser.add_argument("--limit", type=int, default=1000, help="Number of test samples to evaluate (default: 1000 for speed)")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 EVALUATING MODEL ON TEST SPLIT")
    print("=" * 60)

    # 1. Load Dataset
    print(f"Loading dataset from: {args.dataset_dir}...")
    dataset_dict = load_from_disk(args.dataset_dir)
    
    if "test" in dataset_dict:
        test_dataset = dataset_dict["test"]
    elif "validation" in dataset_dict:
        test_dataset = dataset_dict["validation"]
    else:
        test_dataset = dataset_dict
        
    print(f"✅ Loaded test dataset containing {len(test_dataset)} samples.")
    
    if args.limit and args.limit < len(test_dataset):
        test_dataset = test_dataset.select(range(args.limit))
        print(f"ℹ️ Limiting evaluation to first {args.limit} samples.")

    # 2. Load Model and Processor
    print(f"Loading model and processor from: {args.model_dir}...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2PhonemeEmbedder.from_pretrained(args.model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"🖥️ Model loaded on {device}")

    # Build id2phoneme mapping
    vocab = processor.tokenizer.get_vocab()
    id2phoneme = {v: k for k, v in vocab.items()}
    pad_token_id = processor.tokenizer.pad_token_id or 0

    per_scores = []
    
    # 3. Evaluate Loop
    for idx, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):
        try:
            # Get input audio features and labels
            audio = sample["input_values"]
            labels = sample["labels"]
            
            # Collate on the fly (truncate to 20s if needed, like training)
            MAX_AUDIO_SAMPLES = 320000
            if len(audio) > MAX_AUDIO_SAMPLES:
                ratio = MAX_AUDIO_SAMPLES / len(audio)
                audio = audio[:MAX_AUDIO_SAMPLES]
                labels = labels[:max(1, int(len(labels) * ratio))]

            # Convert to tensors
            input_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
            clean_ref = [rid for rid in labels if rid >= 0 and rid != pad_token_id]

            if not clean_ref:
                continue

            # Model Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy().tolist()

            # CTC duplicate collapsing
            collapsed_pred = []
            prev = None
            for pid in pred_ids:
                if pid == prev or pid == pad_token_id:
                    prev = pid
                    continue
                prev = pid
                collapsed_pred.append(pid)

            # Compute PER
            per = calculate_per(clean_ref, collapsed_pred)
            per_scores.append(per)

            # Print detailed logs for the first 3 samples
            if idx < 3:
                ref_str = " ".join([id2phoneme.get(i, f"[{i}]") for i in clean_ref])
                pred_str = " ".join([id2phoneme.get(i, f"[{i}]") for i in collapsed_pred])
                print(f"\n--- Detailed Log: Sample {idx+1} ---")
                print(f"Ref: {ref_str}")
                print(f"Hyp: {pred_str}")
                print(f"PER: {per:.2%}")
                print("-" * 50)

        except Exception as e:
            print(f"Error processing sample index {idx}: {e}")
            continue

    # 4. Report Results
    if per_scores:
        mean_per = np.mean(per_scores)
        median_per = np.median(per_scores)
        std_per = np.std(per_scores)
        
        print("\n" + "="*50)
        print("          OFFLINE TEST DATASET EVALUATION REPORT")
        print("="*50)
        print(f"Successfully Evaluated: {len(per_scores)} samples")
        print("-"*50)
        print(f"Mean PER:    {mean_per:.2%}")
        print(f"Median PER:  {median_per:.2%}")
        print(f"Std Dev PER: {std_per:.2%}")
        print("="*50)
    else:
        print("\n❌ Error: Failed to evaluate any samples.")

if __name__ == "__main__":
    main()

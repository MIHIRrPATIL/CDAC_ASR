import os
import torch
import torch.nn as nn
import json
import argparse
import soundfile as sf
import io
import numpy as w
import traceback
import re
import sys
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor
from phoneme_embedder import Wav2Vec2PhonemeEmbedder
from g2p.g2p_utils import G2PManager
from audio_utils import AudioPreprocessor
from collections import OrderedDict

def calculate_per(reference, hypothesis):
    """Memory-efficient Levenshtein distance for PER."""
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
    # Ensure NLTK resources are available for G2P
    import nltk
    print("Checking NLTK resources...", flush=True)
    for res in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict', 'punkt', 'punkt_tab']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Evaluate Phoneme Embedder on NPTEL dataset")
    parser.add_argument("--model_dir", default="trained_models/20k_steps", help="Path to model directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--skip", type=int, default=50000, help="Skip first N samples")
    parser.add_argument("--sanity_check", action="store_true", help="Run on training data (skip=0) to verify weights")
    args = parser.parse_args()

    if args.sanity_check:
        print("🔍 SANITY CHECK MODE: Reverting skip to 0 to test training data.")
        args.skip = 0

    print(f"Loading model from {args.model_dir}...", flush=True)
    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2PhonemeEmbedder.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Initialize G2P Manager and Audio Preprocessor (SYNC WITH TRAINING)
    print("Initializing components (G2P and AudioPreprocessor)...", flush=True)
    g2p_manager = G2PManager()
    preprocessor_utils = AudioPreprocessor(sr=16000)

    # Load ID to Phoneme mapping
    vocab_path = os.path.join(args.model_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)
    id2phoneme = {v: k for k, v in vocab.items()}
    pad_id = processor.tokenizer.pad_token_id

    print(f"Loading dataset skbose/indian-english-nptel-v0 (streaming)...", flush=True)
    ds = load_dataset("skbose/indian-english-nptel-v0", split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    
    print(f"Skipping {args.skip} samples...", flush=True)
    eval_iterable = ds.skip(args.skip).take(args.num_samples)

    total_per = 0
    count = 0
    iterator = iter(eval_iterable)
    
    print(f"Evaluating {args.num_samples} samples...", flush=True)
    
    for i in range(args.num_samples):
        try:
            # 0. Get Sample
            try:
                sample = next(iterator)
            except (RuntimeError, Exception):
                print(f"\n⚠️ HF Error at step {i}, re-initializing iterator...", flush=True)
                ds_reinit = load_dataset("skbose/indian-english-nptel-v0", split=args.split, streaming=True)
                ds_reinit = ds_reinit.cast_column("audio", Audio(decode=False))
                iterator = iter(ds_reinit.skip(args.skip + i).take(args.num_samples - i))
                sample = next(iterator)

            # 1. Decode audio
            audio_bytes = sample["audio"]["bytes"]
            with io.BytesIO(audio_bytes) as f:
                audio_array, sr = sf.read(f)
            
            # Resample to 16kHz if needed (Matching train_streaming.py)
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            
            # 1.1 SYNC PREPROCESSING: FFT Filter + VAD Trim
            audio_data = preprocessor_utils.preprocess(audio_array)
            
            # Skip extremely long audio
            if len(audio_data) / 16000 > 30:
                continue
            
            # Ensure float32
            audio_data = audio_data.astype(np.float32)
            
            # 2. Preprocess with Processor (Group Norm happens here)
            inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)

            # 3. Inference
            with torch.no_grad():
                outputs = model(input_values=input_values, return_dict=True)
                
                if isinstance(outputs, (dict, OrderedDict)):
                    logits = outputs.get("logits")
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
            # 4. Collapse CTC
            collapsed = []
            prev = None
            unk_count = 0
            for pid in pred_ids:
                if pid != pad_id:
                    if pid == 1: unk_count += 1
                    if pid != prev:
                        collapsed.append(id2phoneme.get(int(pid), "<unk>"))
                prev = pid
            
            # 5. Transcription handling
            trans = sample.get("transcription_normalised") or sample.get("transcription") or ""
            trans = str(trans)
            if not trans.strip(): continue

            target_phonemes = g2p_manager.convert_sentence(trans)

            # 6. PER
            per = calculate_per(target_phonemes, collapsed)
            total_per += per
            count += 1
            
            # Sample display
            if i < 3 or (i % 20 == 0):
                print(f"\n--- Sample {i+1} ---", flush=True)
                print(f"Ref: {' '.join(target_phonemes[:20])}...", flush=True)
                print(f"Hyp: {' '.join(collapsed[:20])}...", flush=True)
                print(f"Stat: {len(pred_ids)} frames, {unk_count} <unk> frames.", flush=True)
                print(f"PER: {per:.2%}", flush=True)
            elif (i+1) % 5 == 0:
                print(f"Processed {i+1}/{args.num_samples}...", end="\r", flush=True)
                
        except StopIteration:
            break
        except Exception as e:
            print(f"Error processing sample {i}: {e}", flush=True)
            continue

    if count > 0:
        avg_per = total_per / count
        print(f"\n\n{'='*40}")
        print(f"FINAL RESULTS: PER = {avg_per:.2%}")
        print(f"{'='*40}")
        if avg_per > 0.8:
            print("⚠️ WARNING: High PER detected. This usually indicates under-training or a vocab mismatch.")
            print("👉 Try running with --sanity_check to see performance on training data.")
    else:
        print("\nNo samples were successfully evaluated.")

if __name__ == "__main__":
    main()

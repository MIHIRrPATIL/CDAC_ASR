import os
import io
import json
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
import Levenshtein

# Import model, processor, and utilities from local project
from transformers import Wav2Vec2Processor
from phoneme_embedder import Wav2Vec2PhonemeEmbedder
from g2p.g2p_utils import G2PManager
from audio_utils import AudioPreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Phoneme Embedder on Indian Accent Dataset")
    parser.add_argument("--dataset_dir", default="indian-accent-dataset", help="Path to the extracted Kaggle dataset")
    parser.add_argument("--model_dir", default="nptel_embedder_checkpoints", help="Path to local model checkpoints or Hugging Face repo")
    parser.add_argument("--processor_dir", default="processor_dir", help="Path to Wav2Vec2 processor directory")
    parser.add_argument("--dict_path", default="g2p/output_full.dict", help="Path to G2P mapping dictionary")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process per split (for quick testing)")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size (default: 1 for simple sequential inference)")
    return parser.parse_args()

def find_audio(speaker_dir):
    """Searches for audio.wav or audio.mp3 inside a speaker directory."""
    for ext in ["wav", "mp3"]:
        path = os.path.join(speaker_dir, f"audio.{ext}")
        if os.path.exists(path):
            return path
    # Fallback to any audio file in the directory
    for f in os.listdir(speaker_dir):
        if f.endswith((".wav", ".mp3")):
            return os.path.join(speaker_dir, f)
    return None

def extract_transcript(speaker_dir):
    """Parses text.json or alignment.txt to get the text transcription."""
    # 1. Try text.json (DeepSpeech output format)
    json_path = os.path.join(speaker_dir, "text.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Format: [{"word": "hello", ...}, ...]
                words = [item.get("word", item.get("text", "")) for item in data]
                words = [w.strip() for w in words if w]
                if words:
                    return " ".join(words)
            elif isinstance(data, dict):
                # Format: {"text": "hello world", ...} or {"words": [...]}
                if "text" in data:
                    return data["text"]
                elif "words" in data and isinstance(data["words"], list):
                    if len(data["words"]) > 0 and isinstance(data["words"][0], dict):
                        words = [w.get("word", "") for w in data["words"]]
                    else:
                        words = data["words"]
                    words = [w.strip() for w in words if w]
                    return " ".join(words)
        except Exception:
            pass

    # 2. Try alignment.txt (Tacotron alignment format)
    align_path = os.path.join(speaker_dir, "alignment.txt")
    if os.path.exists(align_path):
        try:
            with open(align_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Single line of text
            if len(lines) == 1:
                return lines[0].strip()
            
            # Multi-line: Check if it's "start_time end_time word" format
            words = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) >= 2:
                    # Take the last column (the word) if it contains characters
                    word = parts[-1]
                    if any(c.isalpha() for c in word):
                        words.append(word)
                else:
                    words.append(parts[0])
            if words:
                return " ".join(words)
        except Exception:
            pass

    return None

def main():
    args = parse_args()
    
    # Check if dataset directory exists
    if not os.path.exists(args.dataset_dir):
        # Look for it inside the current folder in case it is named differently
        potential_dirs = [d for d in os.listdir(".") if os.path.isdir(d) and "accent" in d.lower()]
        if potential_dirs:
            args.dataset_dir = potential_dirs[0]
            print(f"ℹ️ Provided dataset path not found. Autodetected: {args.dataset_dir}")
        else:
            print(f"❌ Error: Dataset directory '{args.dataset_dir}' not found.")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 1. Load Processor
    print(f"Loading processor from {args.processor_dir}...")
    processor = Wav2Vec2Processor.from_pretrained(args.processor_dir)
    pad_token_id = processor.tokenizer.pad_token_id or 0

    # 2. Load Model
    # Search for latest checkpoint in model_dir
    local_weights = None
    if os.path.exists(args.model_dir):
        checkpoints = sorted(
            [d for d in os.listdir(args.model_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]) if "-" in x else 0
        )
        if checkpoints:
            local_weights = os.path.join(args.model_dir, checkpoints[-1])
            print(f"✅ Found latest checkpoint at: {local_weights}")
        elif os.path.exists(os.path.join(args.model_dir, "model.safetensors")):
            local_weights = args.model_dir
            print(f"✅ Found model weights at root of model_dir: {local_weights}")

    if local_weights:
        print(f"🚀 Loading pre-trained state from {local_weights}...")
        model = Wav2Vec2PhonemeEmbedder.from_pretrained(local_weights)
    else:
        print(f"🚀 Loading model directly from Hugging Face repo ID: {args.model_dir}...")
        model = Wav2Vec2PhonemeEmbedder.from_pretrained(args.model_dir)
    
    model = model.to(device)
    model.eval()

    # 3. Initialize Utilities
    preprocessor = AudioPreprocessor(sr=16000)
    g2p = G2PManager(dict_path=args.dict_path)

    # Crawl Splits
    splits = ["train", "test", "dev"]
    results = {}

    for split in splits:
        split_dir = os.path.join(args.dataset_dir, "audio", split)
        if not os.path.exists(split_dir):
            # Check direct folders without 'audio/' prefix
            split_dir = os.path.join(args.dataset_dir, split)
            if not os.path.exists(split_dir):
                print(f"⚠️ Split folder for '{split}' not found. Skipping.")
                continue

        # Get all speaker folders
        speaker_dirs = [
            os.path.join(split_dir, d) for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ]
        
        if args.limit:
            speaker_dirs = speaker_dirs[:args.limit]

        print(f"\n📊 Evaluating split: {split.upper()} ({len(speaker_dirs)} samples)...")
        
        per_scores = []
        skipped = 0
        error_count = 0
        max_error_prints = 5

        for speaker_dir in tqdm(speaker_dirs):
            audio_path = find_audio(speaker_dir)
            transcript = extract_transcript(speaker_dir)

            if not audio_path or not transcript:
                if error_count < max_error_prints:
                    print(f"⚠️  Skipped {speaker_dir} because audio_path={audio_path} or transcript={'[FOUND]' if transcript else '[NOT FOUND]'}")
                    error_count += 1
                skipped += 1
                continue

            try:
                # 1. Load Audio
                audio_array, sr = librosa.load(audio_path, sr=None)
                if sr != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

                # 2. Preprocess Audio (FFT + VAD)
                clean_audio = preprocessor.preprocess(audio_array)
                if len(clean_audio) == 0:
                    if error_count < max_error_prints:
                        print(f"⚠️  Skipped {speaker_dir} because VAD trimmed it to 0 length")
                        error_count += 1
                    skipped += 1
                    continue

                # 3. Extract Audio Features
                input_values = processor(clean_audio, sampling_rate=16000).input_values[0]
                input_tensor = torch.tensor(input_values, dtype=torch.float32).unsqueeze(0).to(device)

                # 4. G2P conversion of target transcript
                target_phonemes = g2p.convert_sentence(transcript)
                if len(target_phonemes) == 0:
                    if error_count < max_error_prints:
                        print(f"⚠️  Skipped {speaker_dir} because G2P converted sentence to empty phonemes list")
                        error_count += 1
                    skipped += 1
                    continue
                target_ids = processor.tokenizer(target_phonemes, is_split_into_words=True).input_ids
                clean_ref = [rid for rid in target_ids if rid >= 0 and rid != pad_token_id]

                if not clean_ref:
                    if error_count < max_error_prints:
                        print(f"⚠️  Skipped {speaker_dir} because clean tokenized target reference is empty")
                        error_count += 1
                    skipped += 1
                    continue

                # 5. Model Inference
                with torch.no_grad():
                    outputs = model(input_tensor)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy().tolist()

                # 6. Collapse duplicate predictions (CTC decoding)
                collapsed_pred = []
                prev = None
                for pid in pred_ids:
                    if pid == prev or pid == pad_token_id:
                        prev = pid
                        continue
                    prev = pid
                    collapsed_pred.append(pid)

                # 7. Compute Phoneme Error Rate (PER)
                dist = Levenshtein.distance(clean_ref, collapsed_pred)
                max_len = max(len(clean_ref), len(collapsed_pred), 1)
                per = dist / max_len
                per_scores.append(per)

            except Exception as e:
                if error_count < max_error_prints:
                    print(f"⚠️  Error processing {speaker_dir}: {e}")
                    error_count += 1
                skipped += 1
                continue

        if per_scores:
            results[split] = {
                "mean_per": np.mean(per_scores),
                "median_per": np.median(per_scores),
                "std_per": np.std(per_scores),
                "total_processed": len(per_scores),
                "skipped": skipped
            }
            print(f"✅ Split {split.upper()} Results:")
            print(f"   Mean PER:    {results[split]['mean_per']:.2%}")
            print(f"   Median PER:  {results[split]['median_per']:.2%}")
            print(f"   Std Dev PER: {results[split]['std_per']:.2%}")
            print(f"   Processed:   {results[split]['total_processed']} samples (Skipped: {results[split]['skipped']})")
        else:
            print(f"❌ Split {split.upper()} failed to evaluate any samples.")

    # Print Final Summary Table
    if results:
        print("\n" + "="*50)
        print("          FINAL EVALUATION SUMMARY REPORT")
        print("="*50)
        print(f"{'Split':<10} | {'Mean PER':<10} | {'Median PER':<10} | {'Samples':<8}")
        print("-"*50)
        for split, metrics in results.items():
            print(f"{split.upper():<10} | {metrics['mean_per']:.2%}    | {metrics['median_per']:.2%}      | {metrics['total_processed']:<8}")
        print("="*50)

if __name__ == "__main__":
    main()

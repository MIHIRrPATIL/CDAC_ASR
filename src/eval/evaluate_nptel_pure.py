import os
import sys
import json
import argparse
import numpy as np
import librosa
import torch
from tqdm import tqdm
import Levenshtein
import nltk

# Ensure the parent directory and current directory are on sys.path for local imports
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import Wav2Vec2Processor
from src.models.phoneme_embedder import Wav2Vec2PhonemeEmbedder
from src.g2p.g2p_utils import G2PManager
from src.utils.audio_utils import AudioPreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Phoneme Embedder on Local NPTEL-pure Dataset")
    parser.add_argument("--dataset_dir", default="sample_dataset/nptel-pure", help="Path to the local NPTEL-pure dataset")
    parser.add_argument("--model_dir", default="MihirRPatil/nptel-asr-phoneme-v2", help="Hugging Face repo ID or path to local model checkpoints")
    parser.add_argument("--processor_dir", default="models/processor_dir", help="Path to local Wav2Vec2 processor directory")
    parser.add_argument("--dict_path", default="src/g2p/output_v2_detailed.dict", help="Path to G2P mapping dictionary")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process (for quick testing)")
    parser.add_argument("--transcript_mode", choices=["corrected", "original", "metadata"], default="corrected",
                        help="Primary transcript source: human-corrected (corrected), original-ASR (original), or JSON-metadata (metadata)")
    return parser.parse_args()

def get_transcript(dataset_dir, file_hash, transcript_mode="corrected"):
    """
    Retrieves the transcript for a given file hash.
    Tries different modes and falls back sequentially.
    """
    corrected_path = os.path.join(dataset_dir, "corrected_txt", f"{file_hash}.txt")
    original_path = os.path.join(dataset_dir, "original_txt", f"{file_hash}.txt")
    metadata_path = os.path.join(dataset_dir, "metadata", f"{file_hash}.json")

    # Tiered search based on transcript_mode configuration
    if transcript_mode == "corrected":
        search_order = [corrected_path, original_path, metadata_path]
    elif transcript_mode == "original":
        search_order = [original_path, corrected_path, metadata_path]
    else:
        search_order = [metadata_path, corrected_path, original_path]

    for path in search_order:
        if not os.path.exists(path):
            continue
        try:
            if path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        return text
            elif path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    text = data.get("original_phrase", "").strip()
                    if text:
                        return text
        except Exception:
            pass
    return None

def load_processor_and_model(model_dir, processor_dir, device):
    """
    Loads Wav2Vec2Processor and Wav2Vec2PhonemeEmbedder with local path checks and HF fallbacks.
    """
    # 1. Load Processor
    processor = None
    if os.path.exists(processor_dir):
        print(f"Loading processor from local directory: {processor_dir}...")
        try:
            processor = Wav2Vec2Processor.from_pretrained(processor_dir)
        except Exception as e:
            print(f"⚠️ Failed to load processor from local directory {processor_dir}: {e}")
    
    # Fallback to loading processor from the model source
    if processor is None:
        print(f"Trying to load processor from model source: {model_dir}...")
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_dir)
        except Exception as e:
            print(f"⚠️ Failed to load processor from {model_dir}: {e}. Falling back to facebook/wav2vec2-xlsr-53...")
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53")

    # 2. Load Model
    local_weights = None
    if os.path.exists(model_dir):
        checkpoints = sorted(
            [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]) if "-" in x else 0
        )
        if checkpoints:
            local_weights = os.path.join(model_dir, checkpoints[-1])
            print(f"✅ Found latest local checkpoint at: {local_weights}")
        elif os.path.exists(os.path.join(model_dir, "model.safetensors")):
            local_weights = model_dir
            print(f"✅ Found model weights at root of model_dir: {local_weights}")

    if local_weights:
        print(f"🚀 Loading pre-trained state from local path: {local_weights}...")
        model = Wav2Vec2PhonemeEmbedder.from_pretrained(local_weights)
    else:
        print(f"🚀 Loading model directly from HF Hub: {model_dir}...")
        model = Wav2Vec2PhonemeEmbedder.from_pretrained(model_dir)
        
    model = model.to(device)
    model.eval()
    return processor, model

def main():
    args = parse_args()
    
    # Download required NLTK resources
    print("Checking NLTK resources...")
    for res in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict', 'punkt', 'punkt_tab']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

    # Directory sanity checks
    wav_dir = os.path.join(args.dataset_dir, "wav")
    if not os.path.exists(wav_dir):
        print(f"❌ Error: WAV directory '{wav_dir}' not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Load Model and Processor
    processor, model = load_processor_and_model(args.model_dir, args.processor_dir, device)
    pad_token_id = processor.tokenizer.pad_token_id or 0

    # Build token to phoneme mapping for legible output logs
    try:
        vocab = processor.tokenizer.get_vocab()
        id2phoneme = {v: k for k, v in vocab.items()}
    except Exception:
        # Fallback if vocab cannot be extracted
        vocab_json_path = os.path.join(args.model_dir, "vocab.json")
        if os.path.exists(vocab_json_path):
            with open(vocab_json_path, "r", encoding="utf-8") as f:
                v_dict = json.load(f)
            id2phoneme = {v: k for k, v in v_dict.items()}
        else:
            id2phoneme = {}

    # Initialize utilities
    preprocessor = AudioPreprocessor(sr=16000)
    
    # Check if local dict path exists, if not fall back to None to let G2PManager autodetect in its subfolders
    dict_path = args.dict_path
    if not os.path.exists(dict_path):
        print(f"ℹ️ Local G2P dictionary path '{dict_path}' not found. Letting G2PManager auto-detect output_full.dict...")
        dict_path = None
    g2p = G2PManager(dict_path=dict_path)

    # Get all WAV files
    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
    if args.limit:
        wav_files = wav_files[:args.limit]

    print(f"\n📊 Evaluating NPTEL-pure dataset ({len(wav_files)} samples)...")

    per_scores = []
    skipped = 0
    error_count = 0
    max_error_prints = 5

    for filename in tqdm(wav_files):
        file_hash = os.path.splitext(filename)[0]
        audio_path = os.path.join(wav_dir, filename)
        
        # Retrieve transcript with fallbacks
        transcript = get_transcript(args.dataset_dir, file_hash, args.transcript_mode)

        if not transcript:
            if error_count < max_error_prints:
                print(f"⚠️  Skipped {file_hash}: Transcript not found in corrected, original, or metadata files.")
                error_count += 1
            skipped += 1
            continue

        try:
            # 1. Load Audio
            audio_array, sr = librosa.load(audio_path, sr=None)
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

            # 2. Preprocess Audio (Spectral Subtraction + Silero VAD)
            clean_audio = preprocessor.preprocess(audio_array)
            if len(clean_audio) == 0:
                if error_count < max_error_prints:
                    print(f"⚠️  Skipped {file_hash}: VAD trimmed audio to 0 samples.")
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
                    print(f"⚠️  Skipped {file_hash}: G2P conversion resulted in empty phonemes list.")
                    error_count += 1
                skipped += 1
                continue
                
            target_ids = processor.tokenizer.convert_tokens_to_ids(target_phonemes)
            clean_ref = [rid for rid in target_ids if rid >= 0 and rid != pad_token_id]

            if not clean_ref:
                if error_count < max_error_prints:
                    print(f"⚠️  Skipped {file_hash}: Clean tokenized reference is empty.")
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

            # 8. Detailed Comparison Log for the first few samples
            if len(per_scores) <= 3:
                ref_phonemes_str = " ".join([id2phoneme.get(i, f"[{i}]") for i in clean_ref])
                pred_phonemes_str = " ".join([id2phoneme.get(i, f"[{i}]") for i in collapsed_pred])
                print(f"\n--- Detailed Log: Sample {len(per_scores)} ({file_hash}) ---")
                print(f"Transcript: {transcript}")
                print(f"Ref Phonemes: {ref_phonemes_str}")
                print(f"Hyp Phonemes: {pred_phonemes_str}")
                print(f"PER: {per:.2%}")
                print("-" * 50)

        except Exception as e:
            if error_count < max_error_prints:
                print(f"⚠️  Error processing sample {file_hash}: {e}")
                error_count += 1
            skipped += 1
            continue

    # Report Final Statistics
    if per_scores:
        mean_per = np.mean(per_scores)
        median_per = np.median(per_scores)
        std_per = np.std(per_scores)
        total_processed = len(per_scores)
        
        print("\n" + "="*50)
        print("          NPTEL-PURE EVALUATION REPORT")
        print("="*50)
        print(f"Total Files Scanned: {len(wav_files)}")
        print(f"Successfully Processed: {total_processed}")
        print(f"Skipped / Failed:     {skipped}")
        print("-"*50)
        print(f"Mean PER:    {mean_per:.2%}")
        print(f"Median PER:  {median_per:.2%}")
        print(f"Std Dev PER: {std_per:.2%}")
        print("="*50)
    else:
        print("\n❌ Error: Failed to evaluate any samples.")

if __name__ == "__main__":
    main()

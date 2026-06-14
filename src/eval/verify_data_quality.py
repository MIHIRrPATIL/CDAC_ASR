import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
from src.utils.audio_utils import AudioPreprocessor
from evaluate_indian_accent import find_audio

def parse_args():
    parser = argparse.ArgumentParser(description="Check dataset audio quality and VAD behavior")
    parser.add_argument("--dataset_dir", default="indian-accent-dataset/audio", help="Path to Kaggle dataset splits")
    parser.add_argument("--limit", type=int, default=100, help="Number of audio files to analyze (default: 100)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Error: Dataset directory '{args.dataset_dir}' not found.")
        return
        
    preprocessor = AudioPreprocessor(sr=16000)
    
    # Track statistics
    total_scanned = 0
    read_errors = 0
    resample_occurred = 0
    empty_after_vad = 0
    
    original_sample_rates = []
    durations_orig = []
    durations_trimmed = []
    nan_occurred = 0
    
    splits = ["train", "test", "dev"]
    audio_paths = []
    
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
            audio_path = find_audio(sd)
            if audio_path:
                audio_paths.append((sd, audio_path))
                
    print(f"📄 Found {len(audio_paths)} audio files. Analyzing first {args.limit}...")
    
    for speaker_dir, audio_path in tqdm(audio_paths[:args.limit]):
        total_scanned += 1
        try:
            # 1. Try reading audio
            audio_array, sr = librosa.load(audio_path, sr=None)
            original_sample_rates.append(sr)
            
            # Check for NaNs
            if np.isnan(audio_array).any():
                nan_occurred += 1
                
            dur_orig = len(audio_array) / sr
            durations_orig.append(dur_orig)
            
            # 2. Resample check
            if sr != 16000:
                resample_occurred += 1
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                
            # 3. VAD Trimming check
            clean_audio = preprocessor.preprocess(audio_array)
            dur_trimmed = len(clean_audio) / 16000
            durations_trimmed.append(dur_trimmed)
            
            if len(clean_audio) == 0:
                empty_after_vad += 1
                
        except Exception as e:
            read_errors += 1
            print(f"❌ Error reading {audio_path}: {e}")
            
    # Print report
    print("\n" + "="*50)
    print("            AUDIO DATA QUALITY REPORT")
    print("="*50)
    print(f"Total Files Scanned:       {total_scanned}")
    print(f"Read Errors (MP3 backend):  {read_errors} ({read_errors/total_scanned:.2%})")
    print(f"Resampling Needed (!=16kHz): {resample_occurred} ({resample_occurred/total_scanned:.2%})")
    print(f"Empty after VAD Trim:      {empty_after_vad} ({empty_after_vad/total_scanned:.2%})")
    print(f"NaN Values Detected:       {nan_occurred}")
    
    if original_sample_rates:
        print("\n📈 Sample Rate Distribution:")
        rates, counts = np.unique(original_sample_rates, return_counts=True)
        for r, c in zip(rates, counts):
            print(f"  {r} Hz: {c} files")
            
    if durations_orig:
        print(f"\n⏱️  Duration Statistics (Seconds):")
        print(f"  Original Duration: Mean={np.mean(durations_orig):.2f}s, Max={np.max(durations_orig):.2f}s, Min={np.min(durations_orig):.2f}s")
        print(f"  VAD Trimmed Duration: Mean={np.mean(durations_trimmed):.2f}s, Max={np.max(durations_trimmed):.2f}s, Min={np.min(durations_trimmed):.2f}s")
    print("="*50)

if __name__ == "__main__":
    main()

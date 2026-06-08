import os
import io
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import nltk
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor
from g2p.g2p_utils import G2PManager
from audio_utils import AudioPreprocessor

# Global variables to cache model/processor/G2P manager instances per worker process
PREPROCESSOR = None
PROCESSOR = None
G2P_MANAGER = None

def init_worker(processor_dir, dict_path):
    global PREPROCESSOR, PROCESSOR, G2P_MANAGER
    if PREPROCESSOR is None:
        # Prevent PyTorch core oversubscription across multi-process workers
        torch.set_num_threads(1)
        PREPROCESSOR = AudioPreprocessor(sr=16000)
    if PROCESSOR is None:
        PROCESSOR = Wav2Vec2Processor.from_pretrained(processor_dir)
    if G2P_MANAGER is None:
        G2P_MANAGER = G2PManager(dict_path=dict_path)

def preprocess_batch(batch, processor_dir, dict_path):
    # Initialize worker-local resources if not already done
    init_worker(processor_dir, dict_path)
    
    input_values_list = []
    labels_list = []
    
    audios = batch["audio"]
    
    # Find the text column dynamically
    text_key = None
    for key in ["text", "transcription", "sentence", "normalized_text"]:
        if key in batch:
            text_key = key
            break
            
    texts = batch[text_key] if text_key is not None else [""] * len(audios)
                
    for i in range(len(audios)):
        try:
            audio_data = audios[i]
            text = texts[i] if i < len(texts) else ""
            
            # 1. Decode audio bytes
            if isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"] is not None:
                audio_array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            elif isinstance(audio_data, dict) and "array" in audio_data and audio_data["array"] is not None:
                audio_array = np.array(audio_data["array"])
                sr = audio_data.get("sampling_rate", 16000)
            elif isinstance(audio_data, dict) and "path" in audio_data and audio_data["path"] is not None:
                audio_array, sr = sf.read(audio_data["path"])
            else:
                raise ValueError("Invalid audio format or missing audio content.")

            # 2. Resample to 16kHz if needed
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

            # 3. Preprocess Audio (FFT Filter + VAD Trim)
            clean_audio = PREPROCESSOR.preprocess(audio_array)
            if len(clean_audio) == 0:
                raise ValueError("Audio clip is empty after FFT filtering and VAD silence trimming.")

            # 4. Extract Features
            input_values = PROCESSOR(clean_audio, sampling_rate=16000).input_values[0]

            # 5. Text to Phonemes
            phonemes = G2P_MANAGER.convert_sentence(text)
            if len(phonemes) == 0:
                raise ValueError("Phoneme sequence is empty after G2P conversion.")

            # 6. Phonemes to IDs
            labels = PROCESSOR.tokenizer(phonemes, is_split_into_words=True).input_ids
            
            input_values_list.append(input_values)
            labels_list.append(labels)
            
        except Exception as e:
            # We can log exceptions, but to avoid spam, we pass silently.
            pass
            
    return {"input_values": input_values_list, "labels": labels_list}

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess NPTEL dataset offline")
    parser.add_argument("--processor_dir", default="processor_dir", help="Path to local processor config")
    parser.add_argument("--dict_path", default="g2p/output_full.dict", help="Path to MFA dictionary for G2P")
    parser.add_argument("--save_dir", default="local_nptel_processed", help="Path to save the preprocessed dataset")
    parser.add_argument("--num_proc", type=int, default=40, help="Number of processes to use for preprocessing (default: 40)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset to first N samples (for testing/dry runs)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for map function")
    parser.add_argument("--dataset_name", default="skbose/indian-english-nptel-v0", help="Dataset name on Hugging Face Hub")
    args = parser.parse_args()

    print("Checking NLTK resources...")
    for res in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']:
        nltk.download(res, quiet=True)

    # VAD Warmup sequentially in main process before spawning child processes
    print("Warming up Silero VAD cache sequentially in main process...")
    _ = AudioPreprocessor(sr=16000)

    if args.max_samples is not None:
        print(f"Loading first {args.max_samples} samples via streaming to avoid downloading the entire 120 GB...")
        dataset = load_dataset(args.dataset_name, split="train", streaming=True)
        # Cast column to disable auto decoding so we do manual decoding in processes
        dataset = dataset.cast_column("audio", Audio(decode=False))
        
        # Take max_samples
        dataset_stream = dataset.take(args.max_samples)
        
        # Convert to standard in-memory Dataset
        print("Fetching samples from stream...")
        samples = []
        for i, sample in enumerate(dataset_stream):
            samples.append(sample)
            if (i + 1) % 250 == 0:
                print(f"  Loaded {i+1} samples...")
        
        from datasets import Dataset
        dataset = Dataset.from_list(samples)
        print(f"✓ Created in-memory Dataset with {len(dataset)} samples.")
    else:
        print(f"Loading full dataset '{args.dataset_name}' (streaming=False to download to disk)...")
        dataset = load_dataset(args.dataset_name, split="train", streaming=False)
        # Cast column to disable auto decoding so we do manual decoding in processes
        dataset = dataset.cast_column("audio", Audio(decode=False))

    print(f"Starting preprocessing map with {args.num_proc} processes, batch_size={args.batch_size}...")
    
    # We remove all original columns so only the preprocessed input_values and labels are kept
    original_columns = dataset.column_names
    
    processed_dataset = dataset.map(
        preprocess_batch,
        fn_kwargs={"processor_dir": args.processor_dir, "dict_path": args.dict_path},
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=original_columns,
        desc="Preprocessing audio and text offline"
    )

    print(f"Saving preprocessed dataset to disk at '{args.save_dir}'...")
    processed_dataset.save_to_disk(args.save_dir)
    print("✅ Preprocessing and save completed successfully!")

if __name__ == "__main__":
    main()

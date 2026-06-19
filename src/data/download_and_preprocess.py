import os
import sys
import io
import re
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import nltk

# Add the project root to sys.path so we can run the script from any directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets, Dataset
from transformers import Wav2Vec2Processor
from src.g2p.g2p_utils import G2PManager
from src.utils.audio_utils import AudioPreprocessor

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
                try:
                    from scipy.signal import resample_poly
                    import math
                    gcd = math.gcd(sr, 16000)
                    up = 16000 // gcd
                    down = sr // gcd
                    audio_array = resample_poly(audio_array, up, down)
                except Exception:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000, res_type="kaiser_fast")

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

            labels = PROCESSOR.tokenizer.convert_tokens_to_ids(phonemes)
            
            input_values_list.append(input_values)
            labels_list.append(labels)
            
        except Exception as e:
            # We can log exceptions, but to avoid spam, we pass silently.
            pass
            
    return {"input_values": input_values_list, "labels": labels_list}

def is_valid_english_script(text):
    """Exclude non-English script (e.g. Devanagari, Bengali ranges). Only Latin ASCII script is valid."""
    if not text:
        return False
    # Ensure text is purely Latin/ASCII script and spaces/punctuations
    try:
        text.encode('ascii')
        # Check that it contains at least some alphanumeric characters
        return bool(re.search(r"[A-Za-z]", text))
    except UnicodeEncodeError:
        return False

_VOCAB_CACHE = None

def lexical_filter(text, g2p_manager, tokenizer):
    """Verify that all tokens exist in dictionary or neural fallback. Drop if all map to <unk>."""
    global _VOCAB_CACHE
    words = g2p_manager.tokenize(text)
    if not words:
        return False
    
    if _VOCAB_CACHE is None:
        _VOCAB_CACHE = tokenizer.get_vocab()
    vocab = _VOCAB_CACHE
    valid_words = 0
    
    for word in words:
        phonemes = g2p_manager.convert_word(word)
        if len(phonemes) == 0:
            continue
        # If all phonemes for the word map to <unk> (not in vocabulary), skip it
        if all(p not in vocab for p in phonemes):
            continue
        valid_words += 1
        
    return valid_words > 0

def load_mixed_dataset(processor, g2p_manager, token=None, local_openslr_dir="local_openslr_104"):
    """Loads and samples the datasets dynamically using memory-mapped Arrow tables (0-RAM)."""
    
    datasets_list = []
    
    # Helper function to load, filter, and standardize a dataset using Arrow operations (0-RAM)
    def load_and_standardize(path, split, name=None, config=None, text_keys=None, source_label=None, trust_remote_code=False):
        print(f"Loading {path} (config={config}, split={split})...")
        try:
            if config:
                ds = load_dataset(path, config, split=split, token=token, trust_remote_code=trust_remote_code)
            else:
                ds = load_dataset(path, split=split, token=token, trust_remote_code=trust_remote_code)
            
            # Prevent audio decoding into memory
            ds = ds.cast_column("audio", Audio(decode=False))
            
            def filter_fn(example):
                text = ""
                if text_keys:
                    for k in text_keys:
                        if example.get(k):
                            text = example[k]
                            break
                if not text:
                    text = example.get("sentence") or example.get("text") or example.get("transcription") or example.get("normalized_text") or ""
                text = str(text).strip()
                return is_valid_english_script(text) and lexical_filter(text, g2p_manager, processor.tokenizer)

            ds_filtered = ds.filter(filter_fn, desc=f"Filtering {source_label}")
            
            def map_fn(example):
                text = ""
                if text_keys:
                    for k in text_keys:
                        if example.get(k):
                            text = example[k]
                            break
                if not text:
                    text = example.get("sentence") or example.get("text") or example.get("transcription") or example.get("normalized_text") or ""
                return {
                    "audio": example["audio"],
                    "text": str(text).strip(),
                    "source_dataset": source_label
                }
                
            columns_to_remove = [col for col in ds_filtered.column_names if col not in ["audio", "text", "source_dataset"]]
            ds_standardized = ds_filtered.map(
                map_fn,
                remove_columns=columns_to_remove,
                desc=f"Standardizing {source_label}"
            )
            
            datasets_list.append(ds_standardized)
            print(f"✓ Loaded and standardized {len(ds_standardized)} samples from {path}.")
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")

    # 1. Common Voice India Accent
    load_and_standardize("WillHeld/india_accent_cv", "train", text_keys=["sentence"], source_label="common_voice")

    # 2. theothertom/indian_english_extended
    load_and_standardize("theothertom/indian_english_extended", "train", text_keys=["transcription", "sentence"], source_label="theothertom_extended")

    # 3. theothertom/indian_english_bigger
    load_and_standardize("theothertom/indian_english_bigger", "train", text_keys=["transcription", "sentence"], source_label="theothertom_bigger")

    # 4. theothertom/indian_english_audio_2
    load_and_standardize("theothertom/indian_english_audio_2", "train", text_keys=["transcription", "sentence"], source_label="theothertom_audio_2")

    # 5. Svarah (loads 'test' split because 'train' split doesn't exist)
    load_and_standardize("ai4bharat/Svarah", "test", text_keys=["transcription"], source_label="svarah")

    # 6. OpenSLR 104 (bypasses loading script error if local folder exists)
    local_openslr_loaded = False
    if os.path.exists(local_openslr_dir):
        print(f"Loading local OpenSLR 104 from disk ('{local_openslr_dir}')...")
        try:
            local_ds = load_from_disk(local_openslr_dir)
            local_ds = local_ds.cast_column("audio", Audio(decode=False))
            
            def filter_fn(example):
                text = example.get("sentence") or example.get("text") or example.get("transcription") or ""
                text = str(text).strip()
                return is_valid_english_script(text) and lexical_filter(text, g2p_manager, processor.tokenizer)

            ds_filtered = local_ds.filter(filter_fn, desc="Filtering openslr_104")
            
            def map_fn(example):
                text = example.get("sentence") or example.get("text") or example.get("transcription") or ""
                return {
                    "audio": example["audio"],
                    "text": str(text).strip(),
                    "source_dataset": "openslr_104"
                }
                
            columns_to_remove = [col for col in ds_filtered.column_names if col not in ["audio", "text", "source_dataset"]]
            ds_standardized = ds_filtered.map(map_fn, remove_columns=columns_to_remove, desc="Standardizing openslr_104")
            datasets_list.append(ds_standardized)
            local_openslr_loaded = True
            print(f"✓ Loaded local OpenSLR 104: {len(ds_standardized)} samples.")
        except Exception as e:
            print(f"⚠️ Error loading local OpenSLR 104: {e}")
            
    if not local_openslr_loaded:
        load_and_standardize("openslr", "train", config="104", text_keys=["transcription"], source_label="openslr_104", trust_remote_code=True)

    # 7. Eka Care (Medical ASR)
    print("Loading Eka Care Medical ASR...")
    try:
        eka_ds = load_dataset("eka-care/medical-asr", split="train", token=token)
        eka_ds = eka_ds.filter(lambda x: not x.get("is_synthetic", False))
        eka_ds = eka_ds.cast_column("audio", Audio(decode=False))
        
        def filter_fn(example):
            text = example.get("transcription") or example.get("text") or ""
            text = str(text).strip()
            return is_valid_english_script(text) and lexical_filter(text, g2p_manager, processor.tokenizer)

        ds_filtered = eka_ds.filter(filter_fn, desc="Filtering eka_care")
        
        def map_fn(example):
            text = example.get("transcription") or example.get("text") or ""
            return {
                "audio": example["audio"],
                "text": str(text).strip(),
                "source_dataset": "eka_care"
            }
            
        columns_to_remove = [col for col in ds_filtered.column_names if col not in ["audio", "text", "source_dataset"]]
        ds_standardized = ds_filtered.map(map_fn, remove_columns=columns_to_remove, desc="Standardizing eka_care")
        datasets_list.append(ds_standardized)
        print(f"✓ Loaded Eka Care: {len(ds_standardized)} samples.")
    except Exception as e:
        print(f"⚠️ Error loading Eka Care: {e}")

    if not datasets_list:
        raise ValueError("No non-NPTEL datasets were loaded successfully!")

    other_datasets = concatenate_datasets(datasets_list)
    n_others = len(other_datasets)
    print(f"Total non-NPTEL samples gathered: {n_others}")
    print(f"Streaming exactly {n_others} samples from NPTEL to balance...")

    try:
        # Load NPTEL in streaming mode
        nptel_ds = load_dataset("skbose/indian-english-nptel-v0", split="train", streaming=True, token=token)
        nptel_ds = nptel_ds.cast_column("audio", Audio(decode=False))
        
        nptel_samples = []
        loaded = 0
        checked = 0
        for sample in nptel_ds:
            checked += 1
            if checked % 1000 == 0:
                print(f"   [NPTEL Stream] Processed {checked} stream records, matched and balanced {loaded}/{n_others} samples...", flush=True)
            
            text = sample.get("text") or sample.get("transcription") or ""
            text = str(text).strip()
            if is_valid_english_script(text) and lexical_filter(text, g2p_manager, processor.tokenizer):
                nptel_samples.append({
                    "audio": sample["audio"],
                    "text": text,
                    "source_dataset": "nptel"
                })
                loaded += 1
                if loaded >= n_others:
                    break
        print(f"✓ Balanced with {loaded} NPTEL samples (checked {checked} stream items total).")

        nptel_dataset = Dataset.from_list(nptel_samples, features=other_datasets.features)
        final_dataset = concatenate_datasets([other_datasets, nptel_dataset])
    except Exception as e:
        print(f"⚠️ Error loading or processing NPTEL: {e}")
        final_dataset = other_datasets

    print(f"✓ Concatenated and balanced dataset. Total samples: {len(final_dataset)}")
    
    # Shuffle out-of-core
    final_dataset = final_dataset.shuffle(seed=42)
    return final_dataset

def build_and_apply_vocab_patch(dataset, processor, g2p_manager, patch_path):
    """Verify vocabulary against tokenizer. Log any unmapped OOV words to patch_vocab.dict."""
    print("Running G2P vocabulary verification check...")
    unk_id = processor.tokenizer.unk_token_id or 1
    new_patches = {}
    
    # We only check words from non-NPTEL datasets as specified
    words_to_check = set()
    for sample in dataset:
        source = sample.get("source_dataset", "nptel")
        if source != "nptel":
            text = sample.get("text") or sample.get("transcription") or sample.get("sentence") or ""
            words_to_check.update(g2p_manager.tokenize(text))
            
    print(f"Analyzing {len(words_to_check)} unique words from non-NPTEL datasets...")
    
    vocab = processor.tokenizer.get_vocab()
    for word in words_to_check:
        phonemes = g2p_manager.convert_word(word)
        if len(phonemes) == 0:
            continue
        ids = processor.tokenizer.convert_tokens_to_ids(phonemes)
        if any(i == unk_id for i in ids):
            # Clean/approximate phonemes that mapped to unk using valid tokenizer tokens
            cleaned_phonemes = []
            for p in phonemes:
                if p in vocab:
                    cleaned_phonemes.append(p)
                else:
                    closest = ""
                    for char in p:
                        if char in vocab:
                            closest += char
                    if closest:
                        cleaned_phonemes.append(closest)
            
            if cleaned_phonemes:
                new_patches[word] = cleaned_phonemes
                
    if new_patches:
        print(f"Writing {len(new_patches)} new vocabulary patches to {patch_path}...")
        existing_patches = {}
        if os.path.exists(patch_path):
            with open(patch_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        existing_patches[parts[0]] = parts[1].split()
        
        existing_patches.update(new_patches)
        
        os.makedirs(os.path.dirname(patch_path), exist_ok=True)
        with open(patch_path, "w", encoding="utf-8") as f:
            for w, phs in sorted(existing_patches.items()):
                f.write(f"{w}\t{' '.join(phs)}\n")
        
        g2p_manager.phoneme_dict.update(new_patches)
        print("✅ Vocabulary patch successfully updated and merged!")
    else:
        print("✓ No vocabulary patches needed. All words mapped successfully.")

def main():
    parser = argparse.ArgumentParser(description="Download, balance, and preprocess CDAC datasets offline")
    parser.add_argument("--processor_dir", default="models/processor_dir", help="Path to local processor config")
    parser.add_argument("--dict_path", default="src/g2p/output_v2_detailed.dict", help="Path to MFA dictionary for G2P")
    parser.add_argument("--save_dir", default="local_nptel_processed", help="Path to save the preprocessed dataset")
    parser.add_argument("--local_openslr_dir", default="local_openslr_104", help="Path to local processed OpenSLR 104 dataset directory")
    parser.add_argument("--num_proc", type=int, default=40, help="Number of processes to use for preprocessing (default: 40)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for map function")
    parser.add_argument("--hf_token", default=None, help="Hugging Face authorization token")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if isinstance(hf_token, str) and hf_token.strip().lower() in ["none", ""]:
        hf_token = None

    print("Checking NLTK resources...")
    for res in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']:
        nltk.download(res, quiet=True)

    # VAD Warmup sequentially in main process before spawning child processes
    print("Warming up Silero VAD cache sequentially in main process...")
    _ = AudioPreprocessor(sr=16000)

    # Initialize G2P and Processor for verification checks
    processor = Wav2Vec2Processor.from_pretrained(args.processor_dir)
    g2p_manager = G2PManager(dict_path=args.dict_path)

    # 1. Load balanced mixture
    print("Loading datasets dynamically...")
    mixed_dataset = load_mixed_dataset(processor, g2p_manager, token=hf_token, local_openslr_dir=args.local_openslr_dir)

    # 2. Extract and log OOV vocabulary patches
    patch_path = os.path.join(os.path.dirname(args.dict_path), "patch_vocab.dict")
    build_and_apply_vocab_patch(mixed_dataset, processor, g2p_manager, patch_path)

    # 3. Perform Preprocessing Map
    print(f"Starting preprocessing map with {args.num_proc} processes, batch_size={args.batch_size}...")
    original_columns = mixed_dataset.column_names
    
    processed_dataset = mixed_dataset.map(
        preprocess_batch,
        fn_kwargs={"processor_dir": args.processor_dir, "dict_path": args.dict_path},
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=original_columns,
        desc="Preprocessing audio and text offline"
    )

    print(f"Splitting dataset into train and test (10% test size)...")
    dataset_dict = processed_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Saving preprocessed DatasetDict to disk at '{args.save_dir}'...")
    dataset_dict.save_to_disk(args.save_dir)
    print("✅ Preprocessing, train-test split, and save completed successfully!")

if __name__ == "__main__":
    main()

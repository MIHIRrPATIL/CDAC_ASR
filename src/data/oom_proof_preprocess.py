import os
import sys
import io
import re
import shutil
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import nltk
from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets, Dataset, DatasetDict

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import Wav2Vec2Processor
from src.g2p.g2p_utils import G2PManager
from src.utils.audio_utils import AudioPreprocessor

PREPROCESSOR = None
PROCESSOR = None
G2P_MANAGER = None

def init_worker(processor_dir, dict_path):
    global PREPROCESSOR, PROCESSOR, G2P_MANAGER
    if PREPROCESSOR is None:
        torch.set_num_threads(1)
        PREPROCESSOR = AudioPreprocessor(sr=16000)
    if PROCESSOR is None:
        PROCESSOR = Wav2Vec2Processor.from_pretrained(processor_dir)
    if G2P_MANAGER is None:
        G2P_MANAGER = G2PManager(dict_path=dict_path)

def preprocess_batch(batch, processor_dir, dict_path):
    init_worker(processor_dir, dict_path)
    input_values_list = []
    labels_list = []
    audios = batch["audio"]
    
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
            
            if isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"] is not None:
                audio_array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            elif isinstance(audio_data, dict) and "array" in audio_data and audio_data["array"] is not None:
                audio_array = np.array(audio_data["array"])
                sr = audio_data.get("sampling_rate", 16000)
            elif isinstance(audio_data, dict) and "path" in audio_data and audio_data["path"] is not None:
                audio_array, sr = sf.read(audio_data["path"])
            else:
                raise ValueError("Invalid audio format or missing audio content.")

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

            clean_audio = PREPROCESSOR.preprocess(audio_array)
            if len(clean_audio) == 0:
                raise ValueError("Audio clip is empty after FFT filtering and VAD silence trimming.")

            input_values = PROCESSOR(clean_audio, sampling_rate=16000).input_values[0]

            phonemes = G2P_MANAGER.convert_sentence(text)
            if len(phonemes) == 0:
                raise ValueError("Phoneme sequence is empty after G2P conversion.")

            labels = PROCESSOR.tokenizer.convert_tokens_to_ids(phonemes)
            
            input_values_list.append(input_values)
            labels_list.append(labels)
            
        except Exception:
            pass
            
    return {"input_values": input_values_list, "labels": labels_list}

def is_valid_english_script(text):
    if not text:
        return False
    try:
        text.encode('ascii')
        return bool(re.search(r"[A-Za-z]", text))
    except UnicodeEncodeError:
        return False

_VOCAB_CACHE = None

def lexical_filter(text, g2p_manager, tokenizer):
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
        if all(p not in vocab for p in phonemes):
            continue
        valid_words += 1
        
    return valid_words > 0

def build_and_apply_vocab_patch(dataset, processor, g2p_manager, patch_path):
    print("Running G2P vocabulary verification check...")
    unk_id = processor.tokenizer.unk_token_id or 1
    new_patches = {}
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
            cleaned_phonemes = []
            for p in phonemes:
                if p in vocab:
                    cleaned_phonemes.append(p)
                else:
                    closest = "".join([char for char in p if char in vocab])
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

def preprocess_and_save_dataset(ds, text_keys, source_label, save_path, processor_dir, dict_path, num_proc, batch_size, g2p_manager, processor):
    """Processes a single dataset end-to-end and saves it directly to disk (0-RAM footprint)"""
    print(f"\n--- Processing {source_label} ---")
    
    # 1. Cast column to prevent audio loading in RAM
    ds = ds.cast_column("audio", Audio(decode=False))
    
    # 2. Filter
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
    
    # 3. Standardize structure
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
    ds_standardized = ds_filtered.map(map_fn, remove_columns=columns_to_remove, desc=f"Standardizing {source_label}")
    
    # 4. Map to features (audio features + phoneme labels)
    print(f"Running preprocessing map for {source_label} with {num_proc} processes...")
    original_columns = ds_standardized.column_names
    ds_preprocessed = ds_standardized.map(
        preprocess_batch,
        fn_kwargs={"processor_dir": processor_dir, "dict_path": dict_path},
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=original_columns,
        desc=f"Extracting features for {source_label}"
    )
    
    # Save directly to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ds_preprocessed.save_to_disk(save_path)
    print(f"✓ Successfully preprocessed and saved {len(ds_preprocessed)} samples to {save_path}")
    return len(ds_preprocessed)

def main():
    parser = argparse.ArgumentParser(description="OOM-proof Preprocessing Pipeline for CDAC ASR")
    parser.add_argument("--processor_dir", default="models/processor_dir")
    parser.add_argument("--dict_path", default="src/g2p/output_v2_detailed.dict")
    parser.add_argument("--save_dir", default="/data/local_nptel_processed")
    parser.add_argument("--local_openslr_dir", default="/data/local_openslr_104")
    parser.add_argument("--parts_dir", default="/data/preprocessed_parts")
    parser.add_argument("--num_proc", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if isinstance(hf_token, str) and hf_token.strip().lower() in ["none", ""]:
        hf_token = None

    print("Checking NLTK resources...")
    for res in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']:
        nltk.download(res, quiet=True)

    print("Warming up Silero VAD cache...")
    _ = AudioPreprocessor(sr=16000)

    processor = Wav2Vec2Processor.from_pretrained(args.processor_dir)
    g2p_manager = G2PManager(dict_path=args.dict_path)

    # Dictionary of datasets to process
    configs = [
        ("WillHeld/india_accent_cv", "train", ["sentence"], "common_voice", None),
        ("theothertom/indian_english_extended", "train", ["transcription", "sentence"], "theothertom_extended", None),
        ("theothertom/indian_english_bigger", "train", ["transcription", "sentence"], "theothertom_bigger", None),
        ("theothertom/indian_english_audio_2", "train", ["transcription", "sentence"], "theothertom_audio_2", None),
        ("ai4bharat/Svarah", "test", ["transcription"], "svarah", None),
        ("eka-care/medical-asr", "train", ["transcription", "text"], "eka_care", None)
    ]

    parts_counts = {}
    preprocessed_datasets = []

    # 1. Process standard datasets one-by-one (0-RAM OOM protection)
    for path, split, text_keys, label, conf in configs:
        part_save_path = os.path.join(args.parts_dir, label)
        
        # Check if already processed and saved on persistent storage (resume support!)
        if os.path.exists(os.path.join(part_save_path, "dataset_info.json")):
            print(f"✓ Part {label} already preprocessed on disk. Loading...")
            parts_counts[label] = len(load_from_disk(part_save_path))
            preprocessed_datasets.append(part_save_path)
            continue

        try:
            print(f"\nLoading {path}...")
            if conf:
                ds = load_dataset(path, conf, split=split, token=hf_token)
            else:
                ds = load_dataset(path, split=split, token=hf_token)
            
            if label == "eka_care":
                ds = ds.filter(lambda x: not x.get("is_synthetic", False))
                
            count = preprocess_and_save_dataset(
                ds, text_keys, label, part_save_path, 
                args.processor_dir, args.dict_path, args.num_proc, 
                args.batch_size, g2p_manager, processor
            )
            parts_counts[label] = count
            preprocessed_datasets.append(part_save_path)
        except Exception as e:
            print(f"⚠️ Error processing {label}: {e}")

    # 2. Process OpenSLR 104
    openslr_part_path = os.path.join(args.parts_dir, "openslr_104")
    if os.path.exists(os.path.join(openslr_part_path, "dataset_info.json")):
        print("✓ OpenSLR 104 already preprocessed. Loading...")
        parts_counts["openslr_104"] = len(load_from_disk(openslr_part_path))
        preprocessed_datasets.append(openslr_part_path)
    else:
        if os.path.exists(args.local_openslr_dir):
            try:
                print(f"Loading local OpenSLR 104 from {args.local_openslr_dir}...")
                local_ds = load_from_disk(args.local_openslr_dir)
                count = preprocess_and_save_dataset(
                    local_ds, ["transcription", "sentence", "text"], "openslr_104", openslr_part_path,
                    args.processor_dir, args.dict_path, args.num_proc,
                    args.batch_size, g2p_manager, processor
                )
                parts_counts["openslr_104"] = count
                preprocessed_datasets.append(openslr_part_path)
            except Exception as e:
                print(f"⚠️ Error loading OpenSLR 104: {e}")
        else:
            print("⚠️ OpenSLR 104 local directory not found! Skipping OpenSLR.")

    # 3. Sum other datasets to determine NPTEL balance count
    n_others = sum(parts_counts.values())
    print(f"\nTotal non-NPTEL samples processed: {n_others}")

    # 4. Stream and process NPTEL in 5000-sample chunk shards (OOM-proof NPTEL preprocessing)
    nptel_parts_dir = os.path.join(args.parts_dir, "nptel_chunks")
    os.makedirs(nptel_parts_dir, exist_ok=True)
    
    # Let's see how many NPTEL samples we have already processed
    existing_nptel_parts = []
    if os.path.exists(nptel_parts_dir):
        existing_nptel_parts = [os.path.join(nptel_parts_dir, d) for d in os.listdir(nptel_parts_dir) 
                                 if os.path.exists(os.path.join(nptel_parts_dir, d, "dataset_info.json"))]
    
    n_nptel_loaded = sum(len(load_from_disk(p)) for p in existing_nptel_parts)
    print(f"Already preprocessed NPTEL samples found on disk: {n_nptel_loaded}/{n_others}")

    if n_nptel_loaded >= n_others:
        print("✓ NPTEL balancing dataset already fully preprocessed on disk.")
        preprocessed_datasets.extend(existing_nptel_parts)
    else:
        print(f"Streaming remaining NPTEL data from HuggingFace to match {n_others} target...")
        try:
            nptel_ds = load_dataset("skbose/indian-english-nptel-v0", split="train", streaming=True, token=hf_token)
            nptel_ds = nptel_ds.cast_column("audio", Audio(decode=False))
            
            chunk_size = 5000
            current_chunk = []
            chunk_idx = len(existing_nptel_parts)
            loaded = n_nptel_loaded
            checked = 0
            
            # Skip records already gathered in previous run if resuming
            skipped = 0
            
            for sample in nptel_ds:
                checked += 1
                if checked % 1000 == 0:
                    print(f"   [NPTEL Stream] Checked {checked} stream records, matched {loaded + len(current_chunk)}/{n_others}...", flush=True)
                
                text = sample.get("text") or sample.get("transcription") or ""
                text = str(text).strip()
                if is_valid_english_script(text) and lexical_filter(text, g2p_manager, processor.tokenizer):
                    if skipped < n_nptel_loaded:
                        skipped += 1
                        continue
                        
                    current_chunk.append({
                        "audio": sample["audio"],
                        "text": text,
                        "source_dataset": "nptel"
                    })
                    
                    if len(current_chunk) >= chunk_size or (loaded + len(current_chunk)) >= n_others:
                        # Process and save this chunk to disk
                        chunk_ds = Dataset.from_list(current_chunk)
                        chunk_save_path = os.path.join(nptel_parts_dir, f"chunk_{chunk_idx}")
                        
                        original_columns = chunk_ds.column_names
                        print(f"\nProcessing NPTEL shard chunk {chunk_idx} ({len(chunk_ds)} samples)...")
                        chunk_ds_preprocessed = chunk_ds.map(
                            preprocess_batch,
                            fn_kwargs={"processor_dir": args.processor_dir, "dict_path": args.dict_path},
                            batched=True,
                            batch_size=args.batch_size,
                            num_proc=args.num_proc,
                            remove_columns=original_columns,
                            desc=f"Preprocessing NPTEL chunk {chunk_idx}"
                        )
                        chunk_ds_preprocessed.save_to_disk(chunk_save_path)
                        print(f"✓ Saved NPTEL chunk {chunk_idx} to {chunk_save_path}")
                        
                        preprocessed_datasets.append(chunk_save_path)
                        loaded += len(current_chunk)
                        current_chunk = []
                        chunk_idx += 1
                        
                        if loaded >= n_others:
                            break
            
            # Process remaining items in buffer if any
            if current_chunk and loaded < n_others:
                chunk_ds = Dataset.from_list(current_chunk)
                chunk_save_path = os.path.join(nptel_parts_dir, f"chunk_{chunk_idx}")
                original_columns = chunk_ds.column_names
                chunk_ds_preprocessed = chunk_ds.map(
                    preprocess_batch,
                    fn_kwargs={"processor_dir": args.processor_dir, "dict_path": args.dict_path},
                    batched=True,
                    batch_size=args.batch_size,
                    num_proc=args.num_proc,
                    remove_columns=original_columns,
                    desc=f"Preprocessing final NPTEL chunk"
                )
                chunk_ds_preprocessed.save_to_disk(chunk_save_path)
                preprocessed_datasets.append(chunk_save_path)
                loaded += len(current_chunk)
                print(f"✓ Saved final NPTEL chunk to {chunk_save_path}")
                
            print(f"✓ NPTEL preprocessing complete. Balanced with {loaded} NPTEL samples.")
        except Exception as e:
            print(f"⚠️ Error during NPTEL processing: {e}")

    # 5. Concatenate all memory-mapped parts (0-RAM operation)
    print("\n--- Final Dataset Assembly ---")
    print(f"Loading all {len(preprocessed_datasets)} preprocessed partitions from disk...")
    loaded_parts = [load_from_disk(p) for p in preprocessed_datasets]
    
    print("Concatenating all parts...")
    final_dataset = concatenate_datasets(loaded_parts)
    print(f"✓ Concatenated. Total samples: {len(final_dataset)}")
    
    print("Shuffling combined dataset out-of-core...")
    final_dataset = final_dataset.shuffle(seed=42)
    
    print("Splitting dataset into train and test splits (10% test)...")
    dataset_dict = final_dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"Saving final DatasetDict to disk at '{args.save_dir}'...")
    dataset_dict.save_to_disk(args.save_dir)
    print("✅ Preprocessing, train-test split, and save completed successfully!")

    # 6. Clean up temporary parts
    print(f"Cleaning up temporary part files in {args.parts_dir}...")
    try:
        shutil.rmtree(args.parts_dir)
        print("✓ Cleaned up temporary parts directory.")
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")

if __name__ == "__main__":
    main()

import os
import io
import re
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import nltk
from datasets import load_dataset, Audio
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

def lexical_filter(text, g2p_manager, tokenizer):
    """Verify that all tokens exist in dictionary or neural fallback. Drop if all map to <unk>."""
    words = g2p_manager.tokenize(text)
    if not words:
        return False
    
    unk_id = tokenizer.unk_token_id or 1
    valid_words = 0
    
    for word in words:
        phonemes = g2p_manager.convert_word(word)
        if len(phonemes) == 0:
            continue
        ids = tokenizer(phonemes, is_split_into_words=True).input_ids
        if all(i == unk_id for i in ids):
            continue
        valid_words += 1
        
    return valid_words > 0

def load_mixed_dataset(processor, g2p_manager, token=None):
    """Loads and samples the datasets dynamically. 
    It exhaustively loads the 4 non-NPTEL datasets, then loads an equal amount of NPTEL."""
    
    samples = []
    print("Loading 4 non-NPTEL datasets to determine target balanced size...")

    # Load Common Voice (West India accent filter)
    print("Loading Mozilla Common Voice India (West Indian Locale)...")
    try:
        cv_ds = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train", streaming=True, token=token)
        cv_ds = cv_ds.cast_column("audio", Audio(decode=False))
        for sample in cv_ds:
            acc = str(sample.get("accents", sample.get("accent", ""))).lower()
            if "marathi" in acc or "gujarati" in acc or "india" in acc:
                text = sample.get("sentence") or sample.get("text") or ""
                if is_valid_english_script(text) and lexical_filter(text, g2p_manager, processor.tokenizer):
                    sample["source_dataset"] = "common_voice"
                    samples.append(sample)
    except Exception as e:
        print(f"Error loading Mozilla Common Voice: {e}")

    # Load Svarah (North India accent filter)
    print("Loading Svarah...")
    try:
        svarah_ds = load_dataset("ai4bharat/Svarah", split="train", streaming=True)
        svarah_ds = svarah_ds.cast_column("audio", Audio(decode=False))
        for sample in svarah_ds:
            text = sample.get("transcription") or sample.get("text") or ""
            if is_valid_english_script(text) and lexical_filter(text, g2p_manager, processor.tokenizer):
                sample["source_dataset"] = "svarah"
                samples.append(sample)
    except Exception as e:
        print(f"Error loading Svarah: {e}")

    # Load OpenSLR 104 (East India accent split)
    print("Loading OpenSLR 104...")
    try:
        openslr_ds = load_dataset("openslr", "104", split="train", streaming=True)
        openslr_ds = openslr_ds.cast_column("audio", Audio(decode=False))
        for sample in openslr_ds:
            text = sample.get("transcription") or sample.get("text") or ""
            if is_valid_english_script(text) and lexical_filter(text, g2p_manager, processor.tokenizer):
                sample["source_dataset"] = "openslr_104"
                samples.append(sample)
    except Exception as e:
        print(f"Error loading OpenSLR 104: {e}")

    # Load Eka Care (Medical ASR)
    print("Loading Eka Care Medical ASR...")
    try:
        eka_ds = load_dataset("eka-care/medical-asr", split="train", streaming=True, token=token)
        eka_ds = eka_ds.cast_column("audio", Audio(decode=False))
        for sample in eka_ds:
            is_synthetic = sample.get("is_synthetic", False)
            if not is_synthetic:
                text = sample.get("transcription") or sample.get("text") or ""
                if is_valid_english_script(text) and lexical_filter(text, g2p_manager, processor.tokenizer):
                    sample["source_dataset"] = "eka_care"
                    samples.append(sample)
    except Exception as e:
        print(f"Error loading Eka Care: {e}")

    n_others = len(samples)
    print(f"Total non-NPTEL samples gathered: {n_others}")
    print(f"Loading exactly {n_others} samples from NPTEL to balance...")

    try:
        nptel_ds = load_dataset("skbose/indian-english-nptel-v0", split="train", streaming=True)
        nptel_ds = nptel_ds.cast_column("audio", Audio(decode=False))
        nptel_iter = iter(nptel_ds)
        loaded = 0
        while loaded < n_others:
            try:
                sample = next(nptel_iter)
                text = sample.get("text") or sample.get("transcription") or ""
                if is_valid_english_script(text) and lexical_filter(text, g2p_manager, processor.tokenizer):
                    sample["source_dataset"] = "nptel"
                    samples.append(sample)
                    loaded += 1
            except StopIteration:
                print("Reached end of NPTEL stream before matching the count!")
                break
    except Exception as e:
        print(f"Error loading NPTEL: {e}")

    print(f"✓ Concatenated and balanced dataset. Total samples: {len(samples)}")
    # Shuffle dataset
    np.random.shuffle(samples)
    
    from datasets import Dataset
    return Dataset.from_list(samples)

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
        ids = processor.tokenizer(phonemes, is_split_into_words=True).input_ids
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
    parser.add_argument("--processor_dir", default="processor_dir", help="Path to local processor config")
    parser.add_argument("--dict_path", default="g2p/output_full.dict", help="Path to MFA dictionary for G2P")
    parser.add_argument("--save_dir", default="local_nptel_processed", help="Path to save the preprocessed dataset")
    parser.add_argument("--num_proc", type=int, default=40, help="Number of processes to use for preprocessing (default: 40)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for map function")
    parser.add_argument("--hf_token", default=None, help="Hugging Face authorization token")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

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
    mixed_dataset = load_mixed_dataset(processor, g2p_manager, token=hf_token)

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

    print(f"Saving preprocessed dataset to disk at '{args.save_dir}'...")
    processed_dataset.save_to_disk(args.save_dir)
    print("✅ Preprocessing and save completed successfully!")

if __name__ == "__main__":
    main()

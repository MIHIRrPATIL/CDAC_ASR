import os
import io
import torch
import torch.nn as nn
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor, 
    TrainingArguments, 
    Trainer,
    Wav2Vec2Config,
    TrainerCallback
)
try:
    import psutil
except ImportError:
    psutil = None
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import argparse
import json
import re

# Import G2P utility
from g2p.g2p_utils import G2PManager

# Import the custom model and utilities from the local directory
from phoneme_embedder import Wav2Vec2PhonemeEmbedder
from audio_utils import AudioPreprocessor

# G2P logic now handled by G2PManager in g2p/g2p_utils.py

# 2. Data Collator for CTC
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

# 3. System Monitoring Callback
class MonitoringCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        stats = []
        # GPU VRAM
        if torch.cuda.is_available():
            vram = torch.cuda.memory_reserved() / 1024**3
            stats.append(f"VRAM: {vram:.1f}GB")
        
        # System RAM
        if psutil:
            ram = psutil.virtual_memory().percent
            stats.append(f"RAM: {ram}%")
            
        # Disk Space
        st = os.statvfs('/')
        free_disk = (st.f_bavail * st.f_frsize) / 1024**3
        stats.append(f"Disk: {free_disk:.1f}GB free")
        
        if stats:
            print(f"\n📊 SYSTEM: {' | '.join(stats)}")

def main():
    print(f"Current Working Directory: {os.getcwd()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub_model_id", required=True, help="Hugging Face Hub repository ID")
    parser.add_argument("--processor_dir", default="processor_dir", help="Path to local processor config")
    parser.add_argument("--dict_path", default="g2p/output_full.dict", help="Path to MFA dictionary for G2P")
    parser.add_argument("--output_dir", default="nptel_embedder_checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps. Defaults to 4 (normal) or 1 (dry_run).")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--push_hub", action="store_true", help="Push checkpoints to Hugging Face Hub")
    parser.add_argument("--dry_run", action="store_true", help="Perform a quick 5-step test")
    args = parser.parse_args()

    if args.dry_run:
        print("🔧 DRY RUN MODE: Reducing steps to 5 and logging frequently.")
        args.steps = 5
        args.batch_size = 1

    # 1. Load Processor
    print(f"Loading processor from {args.processor_dir}...")
    processor = Wav2Vec2Processor.from_pretrained(args.processor_dir)
    g2p_manager = G2PManager(dict_path=args.dict_path)

    # 2. Load Model (Design A: Embedder)
    print(f"🔍 Checking for weights in: {os.path.abspath(args.output_dir)}")
    
    model_path = "facebook/wav2vec2-base"
    local_weights = None
    
    # Fuzzy Search: If literal path fails, look for anything similar in CWD
    search_dirs = [args.output_dir]
    if not os.path.exists(args.output_dir):
        print(f"⚠️ Literal path {args.output_dir} not found. Searching CWD...")
        all_items = os.listdir(".")
        print(f"📁 CWD Contents: {all_items}")
        for item in all_items:
            if os.path.isdir(item) and "embedder_checkpoints" in item.lower():
                print(f"✨ Found potential match: {item}")
                search_dirs.append(item)

    for s_dir in search_dirs:
        if not os.path.exists(s_dir): continue
        
        # Check root of this dir
        test_path = os.path.join(s_dir, "model.safetensors")
        if os.path.exists(test_path):
            local_weights = test_path
            break
            
        # Check latest checkpoint subfolder
        cpts = sorted([d for d in os.listdir(s_dir) if d.startswith("checkpoint")], 
                      key=lambda x: int(x.split("-")[1]) if "-" in x else 0)
        if cpts:
            test_path = os.path.join(s_dir, cpts[-1], "model.safetensors")
            if os.path.exists(test_path):
                local_weights = test_path
                break

    if local_weights:
        print(f"✅ Found local weights at: {local_weights}")
        model_dir = os.path.dirname(local_weights)
        print(f"🚀 Loading pre-trained state from {model_dir}...")
        model = Wav2Vec2PhonemeEmbedder.from_pretrained(model_dir)
    else:
        print(f"❌ No local weights found. Initializing fresh model from {model_path}...")
        config = Wav2Vec2Config.from_pretrained(model_path)
        config.vocab_size = len(processor.tokenizer)
        config.pad_token_id = processor.tokenizer.pad_token_id
        config.classifier_proj_size = 256
        model = Wav2Vec2PhonemeEmbedder(config)

    # 3. Load NPTEL dataset from HuggingFace Hub (streaming = no disk usage!)
    # We skip the Audio() decoder because it requires torchcodec (incompatible with cu118).
    # Instead, we manually decode audio bytes with soundfile in prepare_dataset.
    print("Loading NPTEL dataset from HuggingFace (streaming)...")
    dataset = load_dataset(
        "skbose/indian-english-nptel-v0",
        split="train",
        streaming=True,
    )
    # CRITICAL: Disable auto-decoding so HF doesn't try to use torchcodec
    dataset = dataset.cast_column("audio", Audio(decode=False))
    print("✓ HuggingFace NPTEL dataset loaded (streaming mode, raw audio bytes).")

    print("Initializing Audio Preprocessor (FFT + Silero VAD)...")
    preprocessor_utils = AudioPreprocessor(sr=16000)

    import soundfile as sf
    import numpy as np

    def prepare_dataset(batch):
        # 1. Decode audio bytes manually (bypass torchcodec requirement)
        audio_data = batch["audio"]
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            # Raw parquet format: {'bytes': b'...', 'path': '...'}
            audio_array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
        elif isinstance(audio_data, dict) and "array" in audio_data:
            # Already decoded (unlikely without torchcodec, but just in case)
            audio_array = np.array(audio_data["array"])
            sr = audio_data.get("sampling_rate", 16000)
        else:
            raise ValueError(f"Unexpected audio format: {type(audio_data)}")
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        
        # 2. Preprocess Audio (FFT Filter + VAD Trim)
        clean_audio = preprocessor_utils.preprocess(audio_array)
        
        # 3. Extract Features
        batch["input_values"] = processor(clean_audio, sampling_rate=16_000).input_values[0]
        
        # 4. Text to Phonemes (uses "text" column from HF dataset)
        text = batch.get("text", batch.get("transcription", ""))
        phonemes = g2p_manager.convert_sentence(text)
        
        # 5. Phonemes to IDs (direct tokenizer call, as_target_processor is removed)
        batch["labels"] = processor.tokenizer(phonemes, is_split_into_words=True).input_ids
            
        return batch

    # Apply preprocessing to stream
    processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

    # 4. Training Arguments (optimized for H100 but compatible with local CPU)
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    
    # Determine Grad Accum
    if args.grad_accum is not None:
        grad_accum_steps = args.grad_accum
    else:
        grad_accum_steps = 1 if args.dry_run else 4

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=0 if args.dry_run else 1000,
        bf16=use_bf16,                         # Use bf16 only if supported (H100)
        fp16=False,                            # Avoid fp16 on CPU
        logging_steps=1 if args.dry_run else 50,
        save_strategy="no" if args.dry_run else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,                    # Keep disk usage < 50GB
        push_to_hub=args.push_hub, 
        hub_model_id=args.hub_model_id,
        report_to="none",
        dataloader_num_workers=0, 
        remove_unused_columns=False,
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        args=training_args,
        train_dataset=processed_dataset,
        callbacks=[MonitoringCallback()],
    )

    # 6. Execute Training
    # Note: We skip resume_from_checkpoint because loading optimizer.pt triggers
    # a security block in transformers for Torch < 2.6.
    # Since we loaded weights manually above, this correctly starts Stage 2.
    print("Starting training loop (Stage 2 / Continued)...")
    trainer.train()

    # Final Save
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    trainer.push_to_hub()

if __name__ == "__main__":
    main()

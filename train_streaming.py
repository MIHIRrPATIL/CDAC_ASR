import os
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
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub_model_id", required=True, help="Hugging Face Hub repository ID")
    parser.add_argument("--processor_dir", default="processor_dir", help="Path to local processor config")
    parser.add_argument("--dict_path", default="g2p/output_full.dict", help="Path to MFA dictionary for G2P")
    parser.add_argument("--output_dir", default="nptel_embedder_checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=1000)
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
    # We use a base config but load into our custom class
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
    config.vocab_size = len(processor.tokenizer)
    config.pad_token_id = processor.tokenizer.pad_token_id
    config.classifier_proj_size = 256 # As per architecture plan
    
    print("Initializing Wav2Vec2PhonemeEmbedder...")
    model = Wav2Vec2PhonemeEmbedder(config)

    # 3. Custom NPTEL Loader using official download scripts
    print(f"Initializing Audio Preprocessor (FFT + Silero VAD)...")
    preprocessor_utils = AudioPreprocessor(sr=16000)

    print(f"Initializing NPTELChunkedLoader using {args.dict_path}...")
    from nptel_loader import NPTELChunkedLoader
    
    train_script = os.path.join("download_scripts", "download_train_data.sh")
    if not os.path.exists(train_script):
        # Fallback if scripts aren't in expected location
        train_script = "download_scripts/download_train_data.sh"

    loader = NPTELChunkedLoader(train_script)
    dataset = loader.get_iterable_dataset()

    def prepare_dataset(batch):
        # 1. Preprocess Audio (FFT Filter + VAD Trim)
        raw_audio = batch["audio"]["array"]
        clean_audio = preprocessor_utils.preprocess(raw_audio)
        
        # 2. Extract Features
        batch["input_values"] = processor(clean_audio, sampling_rate=16_000).input_values[0]
        
        # 3. Text to Phonemes
        phonemes = g2p_manager.convert_sentence(batch["transcription"])
        
        # 4. Phonemes to IDs
        with processor.as_target_processor():
            batch["labels"] = processor(phonemes, is_split_into_words=True).input_ids
            
        return batch

    # Apply preprocessing to stream
    processed_dataset = dataset.map(prepare_dataset)

    # 4. Training Arguments (optimized for H100 but compatible with local CPU)
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1 if args.dry_run else 4,
        learning_rate=args.learning_rate,
        warmup_steps=0 if args.dry_run else 1000,
        bf16=use_bf16,                         # Use bf16 only if supported (H100)
        fp16=False,                            # Avoid fp16 on CPU
        logging_steps=1 if args.dry_run else 50,
        save_strategy="no" if args.dry_run else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,                    # Keep disk usage < 50GB
        push_to_hub=False if args.dry_run else True, # Don't push tests to hub
        hub_model_id=args.hub_model_id,
        report_to="none",
        dataloader_num_workers=0 if args.dry_run else 2, 
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

    # 6. Execute Training (Resume from Hub if possible)
    print("Starting training loop...")
    checkpoint = None
    if os.path.exists(args.output_dir) and any(d.startswith("checkpoint") for d in os.listdir(args.output_dir)):
        checkpoint = True 
        print(f"Resuming from local checkpoint in {args.output_dir}")
    
    trainer.train(resume_from_checkpoint=checkpoint)

    # Final Save
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    trainer.push_to_hub()

if __name__ == "__main__":
    main()

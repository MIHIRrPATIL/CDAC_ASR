import os
import io
import torch
import torch.nn as nn
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
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
import soundfile as sf
import numpy as np

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

# 3. System Monitoring + Early Collapse Detection Callback
class MonitoringCallback(TrainerCallback):
    def __init__(self, model=None, processor=None):
        self.model = model
        self.processor = processor
        self._collapse_warnings = 0
    
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
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Check for collapse every 500 steps (expert recommendation #5)."""
        if state.global_step % 500 != 0 or state.global_step == 0:
            return
        
        m = model or self.model
        if m is None:
            return
        
        try:
            # Sample the phoneme embedding similarities to detect collapse
            with torch.no_grad():
                ph = m.phoneme_embeddings if hasattr(m, 'phoneme_embeddings') else m.module.phoneme_embeddings
                ph_norm = ph / (ph.norm(dim=-1, keepdim=True) + 1e-8)
                sim = torch.matmul(ph_norm, ph_norm.t())
                # Check if embeddings are all too similar (collapse indicator)
                off_diag = sim - torch.eye(sim.size(0), device=sim.device)
                avg_sim = off_diag.abs().mean().item()
                
                if avg_sim > 0.8:
                    self._collapse_warnings += 1
                    print(f"\n⚠️  COLLAPSE WARNING (step {state.global_step}): "
                          f"Phoneme embeddings avg similarity = {avg_sim:.3f} (>0.8). "
                          f"Model may be collapsing! [{self._collapse_warnings} warnings]")
                else:
                    print(f"\n✅ HEALTH CHECK (step {state.global_step}): "
                          f"Phoneme embedding diversity = {1-avg_sim:.3f} (healthy)")
        except Exception:
            pass  # Don't crash training for monitoring

# ── Multi-Threaded Prefetch Dataset ──────────────────────────────────────────
# Replaces the single-threaded .map() with a parallel preprocessing pipeline.
# Each worker thread gets its own Silero VAD instance; the shared processor
# and g2p_manager are GIL-safe (numpy/C calls release the GIL).

class PrefetchDataset(torch.utils.data.IterableDataset):
    """Multi-threaded preprocessing wrapper for HF streaming datasets.
    
    Architecture:
        feeder thread  ──▶  sample_queue  ──▶  N worker threads  ──▶  result_queue  ──▶  DataLoader
    
    Each worker thread owns its own AudioPreprocessor (Silero VAD model)
    to avoid thread-safety issues with VAD internal state.
    """

    _DONE = object()  # Sentinel for worker completion

    def __init__(self, hf_stream, processor, g2p_manager, 
                 num_workers=8, prefetch_size=256):
        super().__init__()
        self.hf_stream = hf_stream
        self.processor = processor
        self.g2p_manager = g2p_manager
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size

    @staticmethod
    def _process_sample(raw, preprocessor, processor, g2p_manager):
        """Preprocess a single raw sample. Runs in a worker thread."""
        # 1. Decode audio bytes manually (bypass torchcodec requirement)
        audio_data = raw["audio"]
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            audio_array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
        elif isinstance(audio_data, dict) and "array" in audio_data:
            audio_array = np.array(audio_data["array"])
            sr = audio_data.get("sampling_rate", 16000)
        else:
            raise ValueError(f"Unexpected audio format: {type(audio_data)}")

        # 2. Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        # 3. Preprocess Audio (FFT Filter + VAD Trim)
        clean_audio = preprocessor.preprocess(audio_array)

        # 4. Extract Features
        input_values = processor(clean_audio, sampling_rate=16_000).input_values[0]

        # 5. Text to Phonemes
        text = raw.get("text", raw.get("transcription", ""))
        phonemes = g2p_manager.convert_sentence(text)

        # 6. Phonemes to IDs
        labels = processor.tokenizer(phonemes, is_split_into_words=True).input_ids

        return {"input_values": input_values, "labels": labels}

    def __iter__(self):
        sample_q = queue.Queue(maxsize=self.prefetch_size * 2)
        result_q = queue.Queue(maxsize=self.prefetch_size)
        skip_count = [0]

        def worker():
            """Worker thread: owns a private AudioPreprocessor + Silero VAD."""
            local_preprocessor = AudioPreprocessor(sr=16000)
            while True:
                raw = sample_q.get()
                if raw is None:  # Poison pill
                    break
                try:
                    processed = self._process_sample(
                        raw, local_preprocessor, self.processor, self.g2p_manager
                    )
                    result_q.put(processed)
                except Exception as e:
                    skip_count[0] += 1
                    if skip_count[0] <= 10 or skip_count[0] % 100 == 0:
                        print(f"⚠️  Skipping sample ({skip_count[0]} total): {e}")
            result_q.put(self._DONE)

        def feeder():
            """Feeder thread: reads from HF stream, feeds sample_q."""
            try:
                for sample in self.hf_stream:
                    sample_q.put(sample)
            finally:
                # Send poison pills to all workers
                for _ in range(self.num_workers):
                    sample_q.put(None)

        # Launch feeder
        feeder_t = threading.Thread(target=feeder, daemon=True, name="prefetch-feeder")
        feeder_t.start()

        # Launch workers
        workers = []
        for i in range(self.num_workers):
            t = threading.Thread(target=worker, daemon=True, name=f"prefetch-worker-{i}")
            t.start()
            workers.append(t)

        print(f"🚀 PrefetchDataset: {self.num_workers} workers, "
              f"prefetch buffer={self.prefetch_size}")

        # Yield results until all workers signal DONE
        done_count = 0
        while done_count < self.num_workers:
            item = result_q.get()
            if item is self._DONE:
                done_count += 1
            else:
                yield item

        if skip_count[0] > 0:
            print(f"📊 PrefetchDataset: skipped {skip_count[0]} bad samples total.")


def main():
    # Ensure NLTK resources used by g2p_en are available
    import nltk
    print("Checking NLTK resources...")
    for res in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']:
        nltk.download(res, quiet=True)
    
    print(f"Current Working Directory: {os.getcwd()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub_model_id", required=True, help="Hugging Face Hub repository ID")
    parser.add_argument("--processor_dir", default="processor_dir", help="Path to local processor config")
    parser.add_argument("--dict_path", default="g2p/output_full.dict", help="Path to MFA dictionary for G2P")
    parser.add_argument("--output_dir", default="nptel_embedder_checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps. Defaults to 2 (normal) or 1 (dry_run).")
    parser.add_argument("--num_workers", type=int, default=12, help="CPU worker threads for preprocessing (default: 12 for multi-core).")
    parser.add_argument("--prefetch", type=int, default=256, help="Prefetch buffer size (preprocessed samples held in RAM).")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
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
    # Instead, we manually decode audio bytes with soundfile in PrefetchDataset workers.
    print("Loading NPTEL dataset from HuggingFace (streaming)...")
    dataset = load_dataset(
        "skbose/indian-english-nptel-v0",
        split="train",
        streaming=True,
    )
    # CRITICAL: Disable auto-decoding so HF doesn't try to use torchcodec
    dataset = dataset.cast_column("audio", Audio(decode=False))
    print("✓ HuggingFace NPTEL dataset loaded (streaming mode, raw audio bytes).")

    # 3b. Wrap in multi-threaded prefetch pipeline
    # Each of the N worker threads gets its own Silero VAD instance.
    # The processor and g2p_manager are shared (GIL-safe for numpy/C calls).
    num_workers = 1 if args.dry_run else args.num_workers
    prefetch = 4 if args.dry_run else args.prefetch
    print(f"Initializing PrefetchDataset ({num_workers} workers, buffer={prefetch})...")
    processed_dataset = PrefetchDataset(
        hf_stream=dataset,
        processor=processor,
        g2p_manager=g2p_manager,
        num_workers=num_workers,
        prefetch_size=prefetch,
    )

    # 4. Training Arguments (optimized for H100 but compatible with local CPU)
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    
    # Determine Grad Accum
    if args.grad_accum is not None:
        grad_accum_steps = args.grad_accum
    else:
        grad_accum_steps = 1 if args.dry_run else 2

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=0 if args.dry_run else 8500,  # 10% of 85k (expert rec)
        max_grad_norm=1.0,                         # Gradient clipping (expert rec)
        bf16=use_bf16,
        fp16=False,
        logging_steps=1 if args.dry_run else 50,
        save_strategy="no" if args.dry_run else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        push_to_hub=args.push_hub, 
        hub_model_id=args.hub_model_id,
        report_to="none",
        dataloader_num_workers=0, 
        remove_unused_columns=False,
    )

    # ── CTC Class Weighting (expert rec: anti-collapse) ──
    # Compute inverse-frequency weights based on the phoneme vocab.
    # Schwa (ə) is the most frequent phoneme in Indian English.
    # This biases the model AWAY from over-predicting common phonemes.
    import json as _json
    vocab_path = os.path.join(args.processor_dir, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r', encoding='utf8') as f:
            vocab = _json.load(f)
        num_classes = len(processor.tokenizer)
        # Heuristic: Schwa gets weight 0.3, blank/unk get 1.0, rest get 1.0
        # We use a simple prior: schwa ~30% of tokens, so downweight it.
        weights = torch.ones(num_classes)
        schwa_ids = [v for k, v in vocab.items() if k in ('ə', 'É™')]
        for sid in schwa_ids:
            weights[sid] = 0.3
        model.ctc_class_weights = weights
        print(f"✅ CTC class weights set: {num_classes} classes, schwa weight=0.3")
    else:
        print(f"⚠️ No vocab.json at {vocab_path}, skipping class weighting.")

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        args=training_args,
        train_dataset=processed_dataset,
        callbacks=[MonitoringCallback(model=model, processor=processor)],
    )

    # 6. Execute Training
    print("Starting training loop (Phase 4: Anti-Collapse)...")
    print(f"  LR: {args.learning_rate}, Warmup: 8500, Grad Clip: 1.0")
    print(f"  Effective Batch: {args.batch_size * grad_accum_steps}")
    trainer.train()

    # Final Save
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if args.push_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    main()

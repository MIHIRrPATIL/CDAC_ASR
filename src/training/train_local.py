import os
import torch
import torch.nn as nn
# Set sharing strategy to file_system to bypass container shared memory (/dev/shm) limitations
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from datasets import load_from_disk
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

# Import the custom model and utilities from the local directory
from src.models.phoneme_embedder import Wav2Vec2PhonemeEmbedder

# Data Collator for CTC
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

# Model Health Check & Real-time Verification Callback
class ModelHealthCheckCallback(TrainerCallback):
    def __init__(self, model=None, processor=None, val_samples=None, dataset=None):
        self.model = model
        self.processor = processor
        self.val_samples = val_samples if val_samples is not None else []
        self.dataset = dataset
        self.consecutive_collapse_count = 0
        self.consecutive_blank_count = 0
        self.consecutive_bad_per_count = 0

    def _save_health_checkpoint(self, model, args, reason):
        print(f"\n🚨 [HEALTH CHECK] CRITICAL: Stopping training due to: {reason}")
        save_path = os.path.join(args.output_dir, "early_stop_health_check")
        print(f"💾 Saving model and processor to {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        
        m_to_save = model.module if hasattr(model, "module") else model
        if hasattr(m_to_save, "save_pretrained"):
            m_to_save.save_pretrained(save_path)
        else:
            torch.save(m_to_save.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
            
        if self.processor is not None:
            self.processor.save_pretrained(save_path)
        print("✅ Model weights successfully preserved!")

    def on_log(self, args, state, control, logs=None, **kwargs):
        stats = []
        if torch.cuda.is_available():
            vram = torch.cuda.memory_reserved() / 1024**3
            stats.append(f"VRAM: {vram:.1f}GB")
        
        if psutil:
            ram = psutil.virtual_memory().percent
            stats.append(f"RAM: {ram}%")
            
        st = os.statvfs('/')
        free_disk = (st.f_bavail * st.f_frsize) / 1024**3
        stats.append(f"Disk: {free_disk:.1f}GB free")
        
        if stats:
            print(f"\n📊 SYSTEM: {' | '.join(stats)}")
            
        # Real-time Loss NaN/Inf Check
        if logs is not None:
            loss = logs.get("loss")
            if loss is not None:
                import math
                loss_val = float(loss)
                if math.isnan(loss_val) or math.isinf(loss_val):
                    self._save_health_checkpoint(kwargs.get("model") or self.model, args, f"NaN or Inf loss detected: {loss_val}")
                    control.should_training_stop = True

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Check model representation diversity and transcription correctness every 500 steps."""
        if state.global_step % 500 != 0 or state.global_step == 0:
            return
        
        m = model or self.model
        if m is None:
            return

        # 1. Phoneme Embedding Similarity Collapse Check
        try:
            with torch.no_grad():
                ph = m.phoneme_embeddings if hasattr(m, 'phoneme_embeddings') else m.module.phoneme_embeddings
                ph_norm = ph / (ph.norm(dim=-1, keepdim=True) + 1e-8)
                sim = torch.matmul(ph_norm, ph_norm.t())
                off_diag = sim - torch.eye(sim.size(0), device=sim.device)
                avg_sim = off_diag.abs().mean().item()
                
                if avg_sim > 0.85:
                    self.consecutive_collapse_count += 1
                    print(f"\n⚠️  COLLAPSE WARNING (step {state.global_step}): "
                          f"Phoneme embeddings avg similarity = {avg_sim:.3f} (>0.85). "
                          f"[{self.consecutive_collapse_count}/2 collapse warnings]")
                    if self.consecutive_collapse_count >= 2:
                        self._save_health_checkpoint(m, args, f"Model collapsed (Avg similarity = {avg_sim:.3f})")
                        control.should_training_stop = True
                        return
                else:
                    self.consecutive_collapse_count = 0
                    print(f"\n✅ HEALTH CHECK (step {state.global_step}): "
                          f"Phoneme embedding diversity = {1-avg_sim:.3f} (healthy)")
        except Exception as e:
            print(f"Warning inside Embedding Collapse checker: {e}")

        # 2. Real-time Output Validation (Phoneme Error Rate & Blank Collapse Checks)
        if self.val_samples:
            try:
                device = m.device if hasattr(m, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                pad_token_id = self.processor.tokenizer.pad_token_id or 0
                unk_token_id = self.processor.tokenizer.unk_token_id or 1
                
                m.eval()
                per_scores = []
                total_blank = 0
                total_unk = 0
                
                # We'll print details for the first 2 samples to avoid spamming the log
                print(f"\n📝 VALIDATION INFERENCE (step {state.global_step}) over {len(self.val_samples)} samples:")
                
                for idx, val_sample in enumerate(self.val_samples):
                    input_values = torch.tensor(val_sample["input_values"], dtype=torch.float32).unsqueeze(0).to(device)
                    ref_ids = val_sample["labels"]
                    
                    with torch.no_grad():
                        outputs = m(input_values)
                        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy().tolist()
                    
                    non_pad_predictions = [pid for pid in pred_ids if pid != pad_token_id]
                    if len(non_pad_predictions) == 0:
                        total_blank += 1
                        
                    # Calculate Phoneme Error Rate (PER) via Levenshtein edit distance
                    collapsed_pred = []
                    prev = None
                    for pid in pred_ids:
                        if pid == prev or pid == pad_token_id:
                            prev = pid
                            continue
                        prev = pid
                        collapsed_pred.append(pid)
                    
                    clean_ref = [rid for rid in ref_ids if rid >= 0 and rid != pad_token_id]
                    
                    import Levenshtein
                    dist = Levenshtein.distance(clean_ref, collapsed_pred)
                    max_len = max(len(clean_ref), len(collapsed_pred), 1)
                    per = dist / max_len
                    per_scores.append(per)
                    
                    unk_count = sum(1 for pid in pred_ids if pid == unk_token_id)
                    total_unk += unk_count
                    
                    if idx < 2:
                        pred_phns = self.processor.tokenizer.convert_ids_to_tokens(collapsed_pred)
                        ref_phns = self.processor.tokenizer.convert_ids_to_tokens(clean_ref)
                        print(f"  [Sample {idx+1}]")
                        print(f"   Target:    {' '.join(ref_phns)}")
                        print(f"   Predicted: {' '.join(pred_phns)}")
                        print(f"   PER: {per:.2%}")
                
                m.train()
                
                mean_per = sum(per_scores) / len(per_scores)
                blank_ratio = total_blank / len(self.val_samples)
                print(f"  [Overall Validation Results]")
                print(f"   Mean PER: {mean_per:.2%}")
                print(f"   Blank samples: {total_blank}/{len(self.val_samples)} ({blank_ratio:.2%})")
                
                # Blank Collapse Check (only active after warmup to prevent false stops during early training)
                warmup_limit = max(5000, int(args.warmup_steps))
                if blank_ratio >= 0.8:
                    if state.global_step > warmup_limit:
                        self.consecutive_blank_count += 1
                        print(f"\n⚠️  BLANK COLLAPSE WARNING (step {state.global_step}): "
                              f"Model is predicting nothing but `<pad>` frames for {blank_ratio:.2%} of samples! "
                              f"[{self.consecutive_blank_count}/2 blank warnings]")
                        if self.consecutive_blank_count >= 2:
                            self._save_health_checkpoint(m, args, "Model output collapsed to 100% silent `<pad>` tokens.")
                            control.should_training_stop = True
                            return
                    else:
                        print(f"\nℹ️  Note (step {state.global_step}): Model is predicting only `<pad>` frames for {blank_ratio:.2%} of samples, "
                              f"which is normal during early warmup phase (step < {warmup_limit}).")
                else:
                    self.consecutive_blank_count = 0
                
                # Zero-<unk> Assertion after warmup
                warmup_limit_unk = max(5000, int(args.warmup_steps))
                if state.global_step > warmup_limit_unk:
                    assert total_unk == 0, f"Assertion failed: predicted {total_unk} <unk> tokens after warmup limit (step {state.global_step})"
                
                # PER Early Stopping (<15% target)
                if mean_per < 0.15:
                    print(f"\n🎉 Validation Mean PER ({mean_per:.2%}) dropped below target threshold of 15%!")
                    self._save_health_checkpoint(m, args, f"Target PER achieved ({mean_per:.2%})")
                    control.should_training_stop = True
                    return
                
                # Divergence check (Mean PER remains at 100% after warmup steps)
                warmup_limit_div = max(10000, int(args.warmup_steps))
                if mean_per >= 0.99 and state.global_step > warmup_limit_div:
                    self.consecutive_bad_per_count += 1
                    print(f"⚠️  DIVERGENCE WARNING (step {state.global_step}): "
                          f"Model has a Mean Phoneme Error Rate of {mean_per:.2%} (>99% mismatch) after warmup. "
                          f"[{self.consecutive_bad_per_count}/3 divergence warnings]")
                    if self.consecutive_bad_per_count >= 3:
                        self._save_health_checkpoint(m, args, f"Model diverged (Mean PER = {mean_per:.2%})")
                        control.should_training_stop = True
                        return
                else:
                    self.consecutive_bad_per_count = 0
                    
            except Exception as e:
                print(f"Warning inside Transcription checker: {e}")


def main():
    print(f"Current Working Directory: {os.getcwd()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_dataset_dir", required=True, help="Directory path of the preprocessed dataset on disk")
    parser.add_argument("--hub_model_id", required=True, help="Hugging Face Hub repository ID")
    parser.add_argument("--processor_dir", default="models/processor_dir", help="Path to local processor config")
    parser.add_argument("--output_dir", default="nptel_embedder_checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps. Defaults to 2 (normal) or 1 (dry_run).")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--push_hub", action="store_true", help="Push checkpoints to Hugging Face Hub")
    parser.add_argument("--dry_run", action="store_true", help="Perform a quick 5-step test")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training dataset to first N samples.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of CPU workers for the PyTorch DataLoader.")
    args = parser.parse_args()

    if args.dry_run:
        print("🔧 DRY RUN MODE: Reducing steps to 5 and logging frequently.")
        args.steps = 5
        args.batch_size = 1

    # 1. Load Processor
    print(f"Loading processor from {args.processor_dir}...")
    processor = Wav2Vec2Processor.from_pretrained(args.processor_dir)

    # 2. Load Model
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

    # 3. Load processed dataset from local disk
    print(f"Loading preprocessed dataset from '{args.offline_dataset_dir}'...")
    processed_dataset = load_from_disk(args.offline_dataset_dir)
    
    if args.max_samples is not None:
        processed_dataset = processed_dataset.select(range(min(args.max_samples, len(processed_dataset))))
        print(f"✓ Restricting training dataset to the first {len(processed_dataset)} samples.")
        
    print(f"✓ Dataset loaded. Total samples: {len(processed_dataset)}")

    # Fetch static validation samples from the preprocessed dataset for real-time health checks
    val_samples_processed = []
    try:
        num_val = min(10, len(processed_dataset))
        for idx in range(num_val):
            val_samples_processed.append(processed_dataset[idx])
        print(f"✅ Loaded {len(val_samples_processed)} validation samples from offline dataset.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load validation samples: {e}")

    # 4. Training Arguments
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
        warmup_steps=0 if args.dry_run else int(0.1 * args.steps),  # 10% of total steps
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
        dataloader_num_workers=args.dataloader_num_workers,                  # Use multiple workers for dataloader to keep GPU saturated
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
        schwa_ids = [v for k, v in vocab.items() if k == 'ə']
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
        callbacks=[ModelHealthCheckCallback(model=model, processor=processor, val_samples=val_samples_processed, dataset=processed_dataset)],
    )

    # 6. Execute Training
    # Check if there is a checkpoint in the output directory to resume from
    resume_checkpoint = None
    if os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_checkpoint = True
            print(f"🔄 Found checkpoints in {args.output_dir}. Resuming training from the latest checkpoint...")

    print("Starting training loop (Phase 4: Anti-Collapse)...")
    print(f"  LR: {args.learning_rate}, Warmup: {training_args.warmup_steps}, Grad Clip: 1.0")
    print(f"  Effective Batch: {args.batch_size * grad_accum_steps}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Final Save
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if args.push_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    main()

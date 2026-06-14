# python train_phoneme_wav2vec2.py --wav_dir D:\Speech-to-text\wave2vec2\nptel-pure\wav --phn_dir phonemes --processor_dir processor_dir --pretrained_model facebook/wav2vec2-large-xlsr-53 --output_dir phoneme_model_out --epochs 5 --batch_size 2


import os, json, argparse
import torchaudio
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    Wav2Vec2Processor
)
from typing import List
import os, glob
import soundfile as sf


def build_examples(wav_dir, phn_dir, max_items=1000):
    data = []
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    
    for i, wav_path in enumerate(wav_files[:max_items]):
        base = os.path.splitext(os.path.basename(wav_path))[0]
        phn_path = os.path.join(phn_dir, base + ".phn")
        if not os.path.exists(phn_path):
            continue

        # Read phonemes
        with open(phn_path, "r", encoding="utf8") as f:
            phonemes = f.read().strip().split()

        # Read audio
        speech, sr = sf.read(wav_path)
        speech = np.array(speech, dtype=np.float32)  # ensure NumPy array

        data.append({
            "id": base,
            "audio": speech,      
            "sampling_rate": sr, 
            "phonemes": phonemes
        })
    print(f"Examples built: {len(data)}")
    return Dataset.from_list(data)


def load_audio(path, target_sr=16000):
    speech, sr = torchaudio.load(path)
    if sr != target_sr:
        speech = torchaudio.transforms.Resample(sr, target_sr)(speech)
    return speech.squeeze().numpy()

from datasets import Audio

def prepare_dataset(dataset, processor, phoneme2id):
    def _process(batch):
        speech = batch["audio"]
        batch["input_values"] = processor(speech, sampling_rate=batch["sampling_rate"]).input_values[0]

        # Convert phonemes to IDs
        batch["labels"] = [phoneme2id.get(p, phoneme2id["<unk>"]) for p in batch["phonemes"]]
        return batch

    return dataset.map(_process)


def data_collator(features, processor):
    # audio arrays are now in "input_values"
    input_values = [f["input_values"] for f in features]
    labels = [f["labels"] for f in features]

    # Pad audio
    batch = processor(input_values, sampling_rate=16000,
                      return_tensors="pt", padding=True)

    # pad labels with -100 for CTC
    max_len = max(len(l) for l in labels)
    labels_padded = [l + [-100]*(max_len - len(l)) for l in labels]
    batch["labels"] = torch.tensor(labels_padded, dtype=torch.long)
    return batch

def levenshtein(a: List[int], b: List[int]) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m]

def compute_metrics(pred):
    logits = pred.predictions
    pred_ids = np.argmax(logits, axis=-1)
    label_ids = pred.label_ids

    pad_id = processor.tokenizer.pad_token_id
    total_edits, total_ref = 0, 0

    for p_ids, l_ids in zip(pred_ids, label_ids):
        p_list, prev = [], None
        for pid in p_ids:
            if pid == prev: continue
            prev = pid
            if pid == pad_id: continue
            p_list.append(int(pid))
        l_list = [int(x) for x in l_ids if x != -100]
        e = levenshtein(l_list, p_list)
        total_edits += e
        total_ref += max(1, len(l_list))

    per = total_edits / total_ref if total_ref > 0 else 0.0
    return {"per": per}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--phn_dir", required=True)
    parser.add_argument("--processor_dir", required=True)
    parser.add_argument("--pretrained_model", default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--output_dir", default="phoneme_wav2vec2_out")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    with open(os.path.join(args.processor_dir, "phoneme2id.json"), "r", encoding="utf8") as fh:
        phoneme2id = json.load(fh)
    vocab_size = len(phoneme2id)

    examples = build_examples(args.wav_dir, args.phn_dir)  
    ds = examples  



    processor = Wav2Vec2Processor.from_pretrained(args.processor_dir)

    ds = prepare_dataset(ds, processor, phoneme2id)

    ds = ds.train_test_split(test_size=0.05)
    train_ds, eval_ds = ds["train"], ds["test"]

    model = Wav2Vec2ForCTC.from_pretrained(args.pretrained_model)
    in_features = model.lm_head.in_features
    model.lm_head = torch.nn.Linear(in_features, vocab_size)
    model.config.vocab_size = vocab_size
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.ctc_zero_infinity = True
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_steps=200,
        logging_steps=50,
        learning_rate=1e-4,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        data_collator=lambda f: data_collator(f, processor),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(" Training finished; model & processor saved to", args.output_dir)

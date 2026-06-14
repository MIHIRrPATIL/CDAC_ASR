# python check.py
from train_phoneme_wav2vec2 import build_examples
from transformers import Wav2Vec2Processor
import os, json

# --- update these paths ---
wav_dir = r"D:\Speech-to-text\wave2vec2\nptel-pure\wav"
phn_dir = r"D:\Speech-to-text\wave2vec2\phonemes"
processor_dir = r"D:\Speech-to-text\wave2vec2\processor_dir"

# Load phoneme mapping
with open(os.path.join(processor_dir, "phoneme2id.json"), "r", encoding="utf8") as fh:
    phoneme2id = json.load(fh)

id2phoneme = {v: k for k, v in phoneme2id.items()}

# Build dataset
ds = build_examples(wav_dir, phn_dir)

# Load processor
processor = Wav2Vec2Processor.from_pretrained(processor_dir)

# Prepare dataset
def prepare_dataset(dataset, processor, phoneme2id):
    def _process(batch):
        # process audio
        batch["input_values"] = processor(batch["audio"], sampling_rate=batch["sampling_rate"]).input_values[0]
        # convert phonemes to label IDs
        batch["labels"] = [phoneme2id.get(p, phoneme2id["<unk>"]) for p in batch["phonemes"]]
        return batch
    return dataset.map(_process)

ds = prepare_dataset(ds, processor, phoneme2id)

import numpy as np

# Show 1 sample
sample = ds[0]
audio_array = np.array(sample["audio"], dtype=np.float32)  # convert back to np.array

print("\n--- Sample ---")
print("ID:", sample["id"])
print("Speech shape:", audio_array.shape)
print("Sampling rate:", sample["sampling_rate"])
print("First 10 phonemes:", sample["phonemes"][:10])
print("First 10 label IDs:", sample["labels"][:10])


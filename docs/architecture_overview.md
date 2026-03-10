# Architecture Overview: Phoneme Pronunciation Correction System

## 1. System Philosophy
The system is designed to provide high-fidelity, phoneme-level pronunciation feedback using a modern embedding-based acoustic model. It prioritizes data efficiency (streaming), session resiliency (checkpoint syncing), and granular feedback (temporal alignment).

## 2. Model Architecture (Design A: Contrastive CTC)
The core model is a modified Wav2Vec2 architecture that uses a **Cosine Similarity Embedding Head** instead of a standard linear classification layer.

### Acoustic Encoder
- **Base:** `Wav2Vec2Model` (e.g., `facebook/wav2vec2-base`).
- **Input:** 16kHz Raw Audio.
- **Output:** Temporal sequence of 768-dimensional hidden states (frames).

### Embedding Head
- **Audio Projection:** Linear layer mapping hidden states to a shared embedding space (e.g., 256 or 768 dimensions).
- **Phoneme Dictionary:** A learnable `nn.Embedding` matrix containing a unique vector for every phoneme in the vocabulary + a CTC blank token.
- **Matching Mechanism:** L2-Normalization followed by Matrix Multiplication (Cosine Similarity).
- **Logit Scaling:** A learnable temperature parameter to adjust the sharpness of the probability distribution.

### Optimization
- **Loss Function:** `CTCLoss` applied to the similarity logits.
- **Rationale:** Learns a robust acoustic-phoneme mapping that generalizes better across diverse accents than simple classification.

---

## 3. Data & Training Pipeline
Engineered to handle the **NPTEL2020** dataset within strict compute/storage constraints.

### Streaming Data Loader
- **Source:** Hugging Face `datasets` in `streaming=True` mode.
- **Workflow:** Pulls shards from the cloud -> Decodes audio -> Preprocesses labels -> Discards raw files immediately.
- **Disk Usage:** Minimal (~constant overhead regardless of dataset size).

### Session Resiliency (24-Hour Loop)
- **Checkpointer:** Integrated with Hugging Face Hub.
- **Rotation:** `save_total_limit=2` to keep disk usage under 50GB.
- **Auto-Sync:** `push_to_hub=True` pushes weights to the cloud in the background.
- **Resume Logic:** Training logic checks for existing checkpoints on the Hub at startup.

---

## 4. Pronunciation Analysis Engine
Translates model predictions into actionable user feedback.

### Temporal Alignment
- **Acoustic Transcription:** The model predicts a sequence of phonemes over time.
- **Reference Mapping:** The target word is converted to phonemes via dictionary lookup.
- **Alignment (Levenshtein/DTW):** Dynamically aligns the prediction against the reference to find:
    - **Matches:** Correct pronunciation.
    - **Substitutions:** Incorrect sound used.
    - **Deletions:** Sound skipped.
    - **Insertions:** Extra sound added.

### Scoring Logic
- **Accuracy:** Percentage of correctly produced phonemes.
- **Duration/Timing:** Comparison of phoneme durations against typical distributions.
- **Feedback:** Visual mapping showing exactly which phonemes were missed or incorrect.

---

## 5. Technology Stack
- **Languages:** Python
- **Frameworks:** PyTorch, Hugging Face (Transformers, Datasets, Accelerate)
- **Acoustic Processing:** Librosa, Sounddevice, Torchaudio
- **Hardware Optimization:** CUDA, `bfloat16` (H100 optimized)

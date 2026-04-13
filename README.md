# Phoneme Pronunciation Correction System

An advanced phoneme-level pronunciation correction system based on **Wav2Vec2** and **Contrastive CTC Alignment**. Designed for high-fidelity feedback and optimized for Indian English (NPTEL2020 dataset).

## 🚀 Key Features
- **Embedding-based Acoustic Model:** Uses cosine similarity for robust phoneme identification.
- **Micro-Diagnostic Feedback:** Identifies precise phoneme-level errors (Substitutions, Deletions, Insertions).
- **Audio Cleaning pipeline:** Integrated FFT spectral subtraction and Silero VAD (Voice Activity Detection).
- **Streaming Training:** Optimized for low-disk environments (50GB limit) using direct Zenodo/HuggingFace streaming.
- **Local Portability:** Easily export GPU-trained models for inference on standard laptop CPUs.

## 📂 Project Structure
- `phoneme_embedder.py`: Core model architecture.
- `train_streaming.py`: Training pipeline with HF Hub integration.
- `test_model.py`: Real-time inference and scoring script.
- `audio_utils.py`: VAD and FFT preprocessing utilities.
- `ScoreCalcs.py`: Phoneme alignment and scoring logic.
- `export_for_local.py`: Script to prepare models for CPU/local use.
- `processor_dir/`: Configuration for the Wav2Vec2 processor and tokenizer.
- `g2p/`: Grapheme-to-Phoneme component (Dictionary, Utilities, Tests).
- `docs/`: Technical reports and implementation details.

## 🛠️ Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
pip install soundfile
python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

### 2. Training (Streaming)
```bash
python train_streaming.py --hub_model_id your-repo/nptel-embedder --batch_size 8 --steps 50000
```

### 3. Testing & Evaluation (Automated)
Run this to calculate the **Phoneme Error Rate (PER)** on a portion of the dataset the model hasn't seen:
```bash
python3 evaluate_model.py --model_dir trained_models/20k_steps --num_samples 100 --skip 50000
```

### 4. Interactive Live Test (Microphone)
Run this for a friendly CLI menu to test with your own voice:
```bash
python3 cli_test_menu.py
```

### 5. Local Export
```bash
python export_for_local.py --checkpoint path_to_checkpoint --output my_local_model
```

## 📚 Documentation
For deeper technical insights, check the `docs/` folder:
- [Architecture Overview](docs/architecture_overview.md)
- [G2P Analysis Report](docs/g2p_analysis_report.md)
- [Implementation Plan](docs/implementation_plan.md)
- [Walkthrough](docs/walkthrough.md)
- [G2P Training & Maintenance Guide](g2p/training_guide.md)

## 📡 Hardware Requirements
- **Training:** Recommended NVIDIA GPU with 24GB+ VRAM (optimized for H100).
- **Inference:** Runs on standard Laptop CPUs or modest Cloud server GPUs.

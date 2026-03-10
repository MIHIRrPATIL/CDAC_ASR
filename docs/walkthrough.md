# Walkthrough: Phoneme Pronunciation Correction System (Design A)

I have successfully implemented the new **Design A (Embedding-based)** architecture for your phoneme pronunciation correction system. This architecture is optimized for your H100 GPU and 50GB storage constraints.

## Changes Implemented

### 1. Core Model: `phoneme_embedder.py`
A custom `Wav2Vec2PhonemeEmbedder` class that replaces the standard linear classification head with a **Cosine Similarity Embedding Head**. This allows for a more robust acoustic-phoneme mapping.

### 4. Custom NPTEL Loader (`nptel_loader.py`)
To satisfy your requirement of using the official download scripts while staying under 50GB:
- It parses your local `download_scripts/` to find the official Zenodo URLs.
- It streams the concatenated parts directly into memory using a custom `ConcatenatedStream`.
- It pairs `.wav` and `.txt` files on the fly and deletes them after yielding, keeping your disk usage at essentially zero.

---

## How to Start Training

1.  **H100 Environment Setup:**
    If you are using a remote H100 (e.g., Lambda, AWS), follow the [GPU Setup Guide](setup_gpu.md) first to ensure CUDA and `libsndfile` are ready.

2.  **Hugging Face Login:**
    ```bash
    hf auth login
    ```
2.  **Verify Download Scripts:**
    Ensure your `download_scripts/` directory contains `download_train_data.sh`. I have already created these for you.

3.  **Local Training Test (Dry Run):**
    Before moving to the H100, you can verify the pipeline works on your laptop (even without a GPU) by running:
    ```bash
    python train_streaming.py --hub_model_id test/dry-run --dry_run
    ```
    This will:
    - Automatically detect your CPU.
    - Run exactly 5 steps.
    - Disable Hub uploading and heavy logging.
    - Use a batch size of 1 to save RAM.

---

## How to use on your Local Device (Laptop)

Once you have a trained checkpoint on the H100, you need to prepare it for your local Windows laptop (where there is no H100).

1.  **Prepare the Local Version:**
    Run this script on the H100 machine after training. It will create a folder with the full-precision weights mapped for CPU use.
    ```bash
    python export_for_local.py --checkpoint nptel_embedder_checkpoints/checkpoint-50000 --output my_local_model
    ```
2.  **Download the Folder:**
    Download the `my_local_model` folder to your laptop.
3.  **Run Inference Locally:**
    The `test_model.py` script on your laptop will automatically detect the lack of a GPU and run the full-precision model on your CPU.
    ```bash
    python test_model.py --model_dir my_local_model --duration 4.0 --word because
    ```
3.  **Resume Training:**
    On your next 24-hour session, simply run the same command. It will detect the local checkpoint or pull the latest one from the Hub.

## How to Test Phoneme Correlation

Once you have a trained model in your `output_dir`:
```bash
python test_model.py --model_dir path_to_your_trained_model --word because
```
The script will now use the **Cosine Similarity logits** to identify phonemes and provide granular feedback via the `PronunciationScorer`.

---

## Next Steps
- [ ] **Monitor Hub Sync:** Ensure your first few checkpoints (every 1000 steps) are successfully uploading to your HF Hub.
- [ ] **Evaluate on OOVs:** Test how the embedding space handles Out-Of-Vocabulary words compared to the old model.

## Reference Documentation
- [Architecture Overview](architecture_overview.md)
- [GPU Setup Guide](setup_gpu.md)
- [G2P Training & Maintenance Guide](../g2p/training_guide.md)

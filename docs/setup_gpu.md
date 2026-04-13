# GPU Environment Setup Guide (H100 / Lambda / RunPod)

Moving from a local Windows Conda environment to a high-performance Linux GPU (like an H100) is a standard transition. Follow these steps to ensure zero environment issues.

## 1. Recommended Environment: Lambda/RunPod/vast.ai
Most H100 providers provide a **PyTorch Docker Image**. **Use this instead of creating a fresh Conda env if possible.** It comes with pre-compiled CUDA kernels and optimized drivers.

## 2. Setup Script (`setup_h100.sh`)
If you are on a fresh Ubuntu machine, run these commands to set up the environment perfectly:

```bash
# 1. Update system
sudo apt-get update && sudo apt-get install -y git-lfs libsndfile1

# 2. Install Python dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall torchcodec -y
pip install soundfile

# 3. Download NLTK data for G2P
python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"

# 4. Authenticate with Hugging Face (Crucial for checkpoint syncing)
huggingface-cli login
```

## 3. Potential Issues & Solutions

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **CUDA Mismatch** | `conda` sometimes installs a private CUDA version that conflicts with the System/Driver version (e.g., cu118). | Use `pip install torch==2.4.0+cu118` inside a simple `venv` without Conda if deploying on instances with older/strict driver requirements. |
| **Pygilstate Crash (torchcodec)** | Datasets auto-uses `torchcodec` which is incompatible with `cu118` resulting in PyGILState release core dumps. | `pip uninstall torchcodec -y` and `pip install soundfile` to manually decode bytes (handled in `train_streaming.py`). |
| **Missing Libraries** | `libsndfile1` is often missing on bare Linux. | Run `sudo apt-get install libsndfile1`. |

## 4. Why your local test was "stuck"
Your local test log showed it was stuck in `ssl.py` / `socket.py`. This is **Normal**. 
- The NPTEL dataset is hosted on Zenodo. 
- The first time you run it, it has to establish a connection and buffer the first few chunks of the `.tar.gz`. 
- On a high-speed GPU server (H100 usually has 1Gbps+ connection), this happens in seconds. On a home laptop, it can take a few minutes.

## 5. Verification on GPU
Once on the H100, run the same dry-run test to verify the GPU is detected:
```bash
python train_streaming.py --hub_model_id test/gpu-verify --dry_run
```
Expected output: `bf16=True`, `no_cuda=False`.

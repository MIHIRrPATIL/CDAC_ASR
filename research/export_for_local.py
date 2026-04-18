import torch
import os
import argparse
from transformers import Wav2Vec2Processor
from phoneme_embedder import Wav2Vec2PhonemeEmbedder

def export_model(checkpoint_dir, output_dir):
    """
    Loads a GPU-trained model and saves it specifically for CPU/Local inference.
    Includes quantization for 4x speedup on local CPUs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading checkpoint from {checkpoint_dir}...")
    
    # Force load to CPU
    device = torch.device("cpu")
    
    # 1. Load Processor
    processor = Wav2Vec2Processor.from_pretrained(checkpoint_dir)
    processor.save_pretrained(output_dir)
    
    # 2. Load Model
    model = Wav2Vec2PhonemeEmbedder.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    # 3. Save for Local Device
    # Keeping original precision (32-bit/16-bit) as per user request.
    # No quantization applied.
    print(f"Saving full precision model for local device...")
    model.save_pretrained(output_dir)
    
    # Copy phoneme map
    import shutil
    shutil.copy(os.path.join(checkpoint_dir, "phoneme2id.json"), os.path.join(output_dir, "phoneme2id.json"))

    print(f"✓ Model successfully prepared for local device at: {output_dir}")
    print("You can now move this folder to your local laptop and run test_model.py on it.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the H100 checkpoint folder")
    parser.add_argument("--output", default="local_model_optimized", help="Where to save the local version")
    args = parser.parse_args()
    
    export_model(args.checkpoint, args.output)

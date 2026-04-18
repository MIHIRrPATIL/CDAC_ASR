import shutil
import os

def selective_zip(source_folder, output_name):
    tmp_dir = "temp_model_export"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 1. Look for the folder
    target = source_folder if os.path.exists(source_folder) else source_folder.lstrip("/")
    if not os.path.exists(target):
        target = "npteL_embedder_checkpoints" # Check capitalization
    
    print(f"🔍 Found folder: {target}")
    print(f"🧹 Copying essential files only (skipping optimizer.pt)...")
    
    # Files needed for inference
    essentials = ["model.safetensors", "config.json", "processor_config.json", 
                  "vocab.json", "tokenizer_config.json", "added_tokens.json"]
    
    for f in essentials:
        src = os.path.join(target, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp_dir, f))
            print(f"  + Added: {f}")

    print(f"📦 Zipping compact model...")
    shutil.make_archive(output_name, 'zip', tmp_dir)
    
    # Cleanup temp dir
    shutil.rmtree(tmp_dir)
    print(f"✅ Success! Created: {output_name}.zip (~380MB)")

if __name__ == "__main__":
    selective_zip("/CDAC/nptel_embedder_checkpoints", "nptel_model_final")

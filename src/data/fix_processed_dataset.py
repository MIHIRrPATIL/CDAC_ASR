import os
import argparse
import shutil
from datasets import load_from_disk

def fix_example(example):
    # Filter out the unk_token_id (1) from labels
    example["labels"] = [label for label in example["labels"] if label != 1]
    return example

def main():
    parser = argparse.ArgumentParser(description="Fix corrupted preprocessed dataset labels in-place")
    parser.add_argument("--dataset_dir", default="/data/local_nptel_processed", help="Path to local processed dataset directory")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' not found.")
        return

    print(f"Loading corrupted dataset from '{args.dataset_dir}'...")
    dataset = load_from_disk(args.dataset_dir)
    
    print("Fixing labels (removing <unk> ID 1)...")
    fixed_dataset = dataset.map(
        fix_example,
        desc="Removing <unk> (ID 1) from label sequences"
    )
    
    temp_save_dir = args.dataset_dir.rstrip("/") + "_fixed_temp"
    print(f"Saving fixed dataset to temporary directory '{temp_save_dir}'...")
    fixed_dataset.save_to_disk(temp_save_dir)
    
    print(f"Replacing '{args.dataset_dir}' with the fixed dataset...")
    shutil.rmtree(args.dataset_dir)
    shutil.move(temp_save_dir, args.dataset_dir)
    
    print("✅ Dataset successfully fixed and updated in-place!")

if __name__ == "__main__":
    main()

import os
import sys
import argparse
import tarfile
import shutil
import requests

try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

try:
    from datasets import Dataset, Audio
    has_datasets = True
except ImportError:
    has_datasets = False

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, stream=True, headers=headers, timeout=60)
        response.raise_for_status()
    except Exception as e:
        print(f"⚠️ Error connecting to {url}: {e}")
        return False
        
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    try:
        with open(dest_path, 'wb') as f:
            if has_tqdm and total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for data in response.iter_content(block_size):
                        f.write(data)
                        pbar.update(len(data))
            else:
                downloaded = 0
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded += len(data)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\rDownloading... {percent:.2f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                    else:
                        sys.stdout.write(f"\rDownloading... {downloaded / (1024*1024):.1f} MB")
                    sys.stdout.flush()
                print()
        print(f"✓ Downloaded successfully: {dest_path}")
        return True
    except Exception as e:
        print(f"⚠️ Error writing file to {dest_path}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def extract_tar(archive_path, extract_to):
    print(f"Extracting {archive_path} to {extract_to}...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            total_members = len(members)
            print(f"Extracting {total_members} files...")
            
            if has_tqdm:
                for member in tqdm(members, desc="Extracting"):
                    tar.extract(member, path=extract_to)
            else:
                for idx, member in enumerate(members):
                    tar.extract(member, path=extract_to)
                    if idx % 1000 == 0:
                        sys.stdout.write(f"\rExtracted {idx}/{total_members} files...")
                        sys.stdout.flush()
                print(f"\rExtracted {total_members}/{total_members} files.")
        print(f"✓ Extracted successfully to {extract_to}")
        return True
    except Exception as e:
        print(f"⚠️ Error extracting archive: {e}")
        return False

def parse_and_create_dataset(extracted_dir, dataset_save_path):
    if not has_datasets:
        print("❌ Error: The 'datasets' library is not installed in the environment. Please run: pip install datasets")
        return False

    print(f"Scanning extracted directory: {extracted_dir}...")
    
    # 1. Find all WAV files recursively
    wav_files = {}
    print("Finding WAV files...")
    for root, dirs, files in os.walk(extracted_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                abs_path = os.path.abspath(os.path.join(root, f))
                name = os.path.splitext(f)[0]
                wav_files[name] = abs_path
                
    print(f"Found {len(wav_files)} WAV files.")
    if not wav_files:
        print("❌ Error: No WAV files found in the extracted directory!")
        return False

    # 2. Look for transcription files (case-insensitive search for transcription or text)
    trans_file = None
    possible_names = ["transcription.txt", "transcriptions.txt", "transcription.tsv", "text"]
    for root, dirs, files in os.walk(extracted_dir):
        for f in files:
            if f.lower() in possible_names:
                trans_file = os.path.join(root, f)
                break
        if trans_file:
            break
            
    if not trans_file:
        print("Looking for any .txt or .tsv files as fallback...")
        for root, dirs, files in os.walk(extracted_dir):
            for f in files:
                if f.endswith(".txt") or f.endswith(".tsv"):
                    path = os.path.join(root, f)
                    try:
                        with open(path, "r", encoding="utf-8") as file:
                            lines = [file.readline() for _ in range(5)]
                            valid_lines = 0
                            for line in lines:
                                if not line.strip():
                                    continue
                                parts = line.strip().split(None, 1)
                                if len(parts) == 2:
                                    valid_lines += 1
                            if valid_lines >= 2:
                                trans_file = path
                                break
                    except Exception:
                        pass
            if trans_file:
                break

    if not trans_file:
        print("❌ Error: Could not find any transcription text file!")
        return False
        
    print(f"Using transcription file: {trans_file}")
    
    # 3. Read and parse transcription file
    samples = []
    unmatched_count = 0
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
                
            audio_id, transcription = parts
            
            # Skip potential header row
            if audio_id.lower() in ["path", "id", "file_name", "audio_id"] and transcription.lower() in ["sentence", "transcription", "text"]:
                continue
                
            audio_key = audio_id
            if audio_key.endswith(".wav"):
                audio_key = audio_key[:-4]
                
            wav_path = wav_files.get(audio_key)
            if not wav_path:
                # Suffix/prefix matching fallback
                for name, path in wav_files.items():
                    if name.endswith(audio_key) or audio_key.endswith(name):
                        wav_path = path
                        break
                        
            if wav_path:
                samples.append({
                    "audio": wav_path,
                    "transcription": transcription.strip()
                })
            else:
                unmatched_count += 1
                if unmatched_count <= 5:
                    print(f"Warning: Could not find WAV file for audio ID: {audio_id}")

    print(f"Mapped {len(samples)} samples. Unmatched: {unmatched_count}")
    if not samples:
        print("❌ Error: No samples were mapped successfully!")
        return False
        
    # 4. Create Hugging Face Dataset and cast to Audio
    print("Creating Hugging Face Dataset...")
    dataset = Dataset.from_list(samples)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    # Save to disk
    print(f"Saving dataset to disk at {dataset_save_path}...")
    dataset.save_to_disk(dataset_save_path)
    print(f"✅ Success! Dataset saved to '{dataset_save_path}'.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download and build local OpenSLR 104 dataset offline")
    parser.add_argument("--save_path", default="local_openslr_104", help="Path to save the processed DatasetDict to disk")
    parser.add_argument("--temp_dir", default="temp_openslr_104", help="Path to temporary directory for downloads/extraction")
    parser.add_argument("--cleanup", action="store_true", help="Clean up temporary extracted files and tarballs after dataset creation")
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)
    
    archive_name = "Hindi-English_train.tar.gz"
    archive_path = os.path.join(args.temp_dir, archive_name)
    extracted_dir = os.path.join(args.temp_dir, "extracted")
    
    # Try downloading from main site, fallback to mirror
    download_urls = [
        f"https://www.openslr.org/resources/104/{archive_name}",
        f"https://openslr.trmal.net/resources/104/{archive_name}"
    ]
    
    success = False
    if os.path.exists(archive_path):
        print(f"Archive already exists at {archive_path}. Skipping download.")
        success = True
    else:
        for url in download_urls:
            if download_file(url, archive_path):
                success = True
                break
            print("Trying fallback download URL...")
            
    if not success:
        print("❌ Error: Failed to download the OpenSLR 104 dataset!")
        sys.exit(1)
        
    # Extract
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir, exist_ok=True)
        if not extract_tar(archive_path, extracted_dir):
            print("❌ Error: Failed to extract the dataset archive!")
            sys.exit(1)
    else:
        print(f"Extracted directory already exists at {extracted_dir}. Skipping extraction.")
        
    # Process
    if parse_and_create_dataset(extracted_dir, args.save_path):
        print("✅ Dataset construction finished successfully!")
        
        # Cleanup if requested
        if args.cleanup:
            print("Cleaning up temporary directories...")
            try:
                shutil.rmtree(extracted_dir)
                os.remove(archive_path)
                # Remove temp_dir if empty
                if not os.listdir(args.temp_dir):
                    os.rmdir(args.temp_dir)
                print("✓ Temporary files cleaned up successfully.")
            except Exception as e:
                print(f"Warning: Cleanup failed: {e}")
    else:
        print("❌ Error: Failed to process the dataset!")
        sys.exit(1)

if __name__ == "__main__":
    main()

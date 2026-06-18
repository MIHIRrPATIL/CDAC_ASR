import os
import sys
import argparse
import tarfile
import shutil
import re
import requests

# Add the project root to sys.path so we can run the script from any directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

def parse_and_create_dataset_simple(extracted_dir, dataset_save_path):
    wav_files = {}
    for root, dirs, files in os.walk(extracted_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                abs_path = os.path.abspath(os.path.join(root, f))
                name = os.path.splitext(f)[0]
                wav_files[name] = abs_path
                
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
        return False

    samples = []
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            audio_id, transcription = parts
            if audio_id.lower() in ["path", "id", "file_name", "audio_id"]:
                continue
            audio_key = audio_id[:-4] if audio_id.endswith(".wav") else audio_id
            wav_path = wav_files.get(audio_key)
            if wav_path:
                samples.append({
                    "audio": wav_path,
                    "transcription": transcription.strip()
                })
    return samples

def parse_kaldi_and_slice(extracted_dir, dataset_save_path):
    if not has_datasets:
        print("❌ Error: The 'datasets' library is not installed in the environment. Please run: pip install datasets")
        return False

    import soundfile as sf
    print("Parsing Kaldi-style data structure for OpenSLR 104...")
    
    # Locate transcripts, segments, and wav.scp
    text_file = None
    segments_file = None
    wav_scp_file = None
    
    for root, dirs, files in os.walk(extracted_dir):
        for f in files:
            path = os.path.join(root, f)
            if f == "text":
                text_file = path
            elif f == "segments":
                segments_file = path
            elif f == "wav.scp":
                wav_scp_file = path
                
    if not text_file:
        print("❌ Error: Could not find 'text' (transcriptions) file!")
        return False
        
    print(f"Using transcription file: {text_file}")
    
    # Fallback to simple parser if Kaldi metadata is missing
    if not segments_file or not wav_scp_file:
        print("Warning: 'segments' or 'wav.scp' not found. Falling back to simple file matcher.")
        samples = parse_and_create_dataset_simple(extracted_dir, dataset_save_path)
        if not samples:
            print("❌ Error: No samples were parsed successfully!")
            return False
    else:
        print(f"Found segments file: {segments_file}")
        print(f"Found wav.scp file: {wav_scp_file}")
        
        # 1. Parse wav.scp
        long_wavs = {}
        with open(wav_scp_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    long_id, path_val = parts
                    path_val = path_val.strip()
                    
                    if path_val.endswith("|"):
                        matches = re.findall(r"[^\s/]+\.wav", path_val)
                        wav_name = matches[-1] if matches else long_id + ".wav"
                    else:
                        wav_name = os.path.basename(path_val)
                    
                    found_path = None
                    for root, dirs, files in os.walk(extracted_dir):
                        if wav_name in files:
                            found_path = os.path.join(root, wav_name)
                            break
                    if not found_path:
                        for root, dirs, files in os.walk(extracted_dir):
                            if f"{long_id}.wav" in files:
                                found_path = os.path.join(root, f"{long_id}.wav")
                                break
                                
                    if found_path:
                        long_wavs[long_id] = os.path.abspath(found_path)
                    else:
                        print(f"Warning: Could not locate audio file for long ID: {long_id} (searched for {wav_name})")
                        
        print(f"Parsed {len(long_wavs)} long-form source WAV files.")
        
        # 2. Parse segments
        segments = {}
        with open(segments_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    seg_id, long_id, start_str, end_str = parts[:4]
                    try:
                        segments[seg_id] = {
                            "long_id": long_id,
                            "start": float(start_str),
                            "end": float(end_str)
                        }
                    except ValueError:
                        pass
                        
        print(f"Parsed {len(segments)} segment definitions.")
        
        # 3. Create directory for sliced WAVs
        sliced_dir = os.path.abspath(os.path.join(dataset_save_path, "wavs"))
        os.makedirs(sliced_dir, exist_ok=True)
        
        # 4. Parse text and slice audio
        samples = []
        skipped_count = 0
        cached_long_audio = {}
        
        with open(text_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        print(f"Processing and slicing {len(lines)} transcript segments...")
        iterator = tqdm(lines, desc="Slicing audio") if has_tqdm else lines
        
        for line in iterator:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            seg_id, transcription = parts
            
            if seg_id.lower() in ["path", "id", "file_name", "audio_id"] and transcription.lower() in ["sentence", "transcription", "text"]:
                continue
                
            seg_def = segments.get(seg_id)
            if not seg_def:
                skipped_count += 1
                continue
                
            long_id = seg_def["long_id"]
            long_path = long_wavs.get(long_id)
            if not long_path:
                skipped_count += 1
                continue
                
            seg_wav_path = os.path.join(sliced_dir, f"{seg_id}.wav")
            
            try:
                if long_path not in cached_long_audio:
                    if len(cached_long_audio) >= 5:
                        cached_long_audio.pop(next(iter(cached_long_audio)))
                    audio_data, sr = sf.read(long_path)
                    cached_long_audio[long_path] = (audio_data, sr)
                else:
                    audio_data, sr = cached_long_audio[long_path]
                    
                start_sample = int(seg_def["start"] * sr)
                end_sample = int(seg_def["end"] * sr)
                
                sliced_data = audio_data[start_sample:end_sample]
                if len(sliced_data) > 0:
                    sf.write(seg_wav_path, sliced_data, sr)
                    samples.append({
                        "audio": seg_wav_path,
                        "transcription": transcription.strip()
                    })
                else:
                    skipped_count += 1
            except Exception as e:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"Error slicing segment {seg_id}: {e}")

        print(f"Successfully sliced and mapped {len(samples)} segments. Skipped/Unmatched: {skipped_count}")

    if not samples:
        print("❌ Error: No samples were mapped successfully!")
        return False
        
    # 5. Create Hugging Face Dataset and save to disk
    print("Creating Hugging Face Dataset...")
    dataset = Dataset.from_list(samples)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
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
        
    # Process using Kaldi slicer
    if parse_kaldi_and_slice(extracted_dir, args.save_path):
        print("✅ Dataset construction finished successfully!")
        
        # Cleanup if requested
        if args.cleanup:
            print("Cleaning up temporary directories...")
            try:
                shutil.rmtree(extracted_dir)
                os.remove(archive_path)
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

import os
import re
import requests
import tarfile
import shutil
from pathlib import Path

class NPTELChunkedLoader:
    """
    A loader that parses official NPTEL download scripts, downloads parts one by one,
    yields the records, and deletes the parts to save disk space.
    """
    def __init__(self, script_path, download_dir="temp_nptel"):
        self.script_path = script_path
        self.download_dir = Path(download_dir)
        self.urls = self._parse_script(script_path)
        
    def _parse_script(self, path):
        urls = []
        with open(path, "r") as f:
            for line in f:
                match = re.search(r"https://zenodo.org/record/\d+/files/[^\s]+", line)
                if match:
                    urls.append(match.group(0))
        return urls

    def _download_file(self, url, dest):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def stream_data(self):
        """Generates audio arrays and transcriptions one part at a time."""
        if not self.download_dir.exists():
            self.download_dir.mkdir(parents=True)

        for url in self.urls:
            filename = url.split("/")[-1]
            part_path = self.download_dir / filename
            extract_path = self.download_dir / "extracted"
            
            try:
                # 1. Download
                self._download_file(url, part_path)
                
                # 2. Extract
                # NPTEL parts are often .tar.gz fragments or individual tars
                # If they are parts of one big tar, we need to handle them carefully.
                # However, Zenodo often hosts them as individual tars in some versions.
                # Based on the script 'cat part* > big.tar', they are FRAGMENTS.
                
                # CRITICAL: Since they are fragments, we can't extract them individually
                # UNLESS we download ALL fragments to concatenate (which exceeds space).
                
                # ALTERNATIVE: Use the Hugging Face Streaming version which handles this 
                # opaque fragmentation internally without disk overhead.
                
                # RE-EVALUATION: The user wants to use the "scripts". 
                # If the scripts require concatenation, and we have 50GB, we can't.
                
                # WAIT: Is there a way to stream a concatenated stream?
                # Yes, we can stream multiple response bodies into a tarfile object.
                
                yield from self._stream_concatenated_tars()
                return # We only need to start the concatenated stream once
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
            finally:
                if part_path.exists():
                    os.remove(part_path)
                if extract_path.exists():
                    shutil.rmtree(extract_path)

    def _stream_concatenated_tars(self):
        """
        Streams all URLs as a single concatenated byte stream and passes it to tarfile.
        This avoids saving the 100GB+ tar to disk.
        """
        import io
        
        class ConcatenatedStream(io.RawIOBase):
            def __init__(self, urls):
                self.urls = urls
                self.current_url_idx = 0
                self.current_resp = None
                self.current_iter = None

            def readinto(self, b):
                if self.current_iter is None:
                    if self.current_url_idx >= len(self.urls):
                        return 0
                    url = self.urls[self.current_url_idx]
                    print(f"Streaming part {self.current_url_idx+1}/{len(self.urls)}: {url}")
                    self.current_resp = requests.get(url, stream=True)
                    self.current_resp.raise_for_status()
                    self.current_iter = self.current_resp.iter_content(chunk_size=len(b))
                
                try:
                    chunk = next(self.current_iter)
                    n = len(chunk)
                    b[:n] = chunk
                    return n
                except StopIteration:
                    self.current_url_idx += 1
                    self.current_iter = None
                    return self.readinto(b)

        stream = ConcatenatedStream(self.urls)
        # mode='r|gz' allows streaming reading of a tar.gz
        # The NPTEL dataset structure is:
        # nptel-train/wav/ID.wav
        # nptel-train/corrected_txt/ID.txt
        
        audio_buffer = {}
        text_buffer = {}

        import soundfile as sf
        import io

        with tarfile.open(fileobj=stream, mode='r|gz') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                
                name = member.name
                if name.endswith(".wav"):
                    # Extract ID from wav/ID.wav
                    sid = os.path.basename(name).replace(".wav", "")
                    f = tar.extractfile(member)
                    if f:
                        # Read audio into memory (small chunks)
                        audio_data, sr = sf.read(io.BytesIO(f.read()))
                        audio_buffer[sid] = {"array": audio_data, "sampling_rate": sr}
                
                elif name.endswith(".txt"):
                    sid = os.path.basename(name).replace(".txt", "")
                    f = tar.extractfile(member)
                    if f:
                        text = f.read().decode("utf-8").strip()
                        text_buffer[sid] = text
                
                # Check for matches in buffer
                common_ids = set(audio_buffer.keys()) & set(text_buffer.keys())
                for sid in list(common_ids):
                    yield {
                        "id": sid,
                        "audio": audio_buffer.pop(sid),
                        "transcription": text_buffer.pop(sid)
                    }

    def get_iterable_dataset(self):
        from datasets import IterableDataset
        return IterableDataset.from_generator(self.stream_data)

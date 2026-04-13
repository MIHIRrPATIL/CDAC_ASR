import os
import re
import requests
import tarfile
import io
import time
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class NPTELChunkedLoader:
    """
    A loader for the NPTEL2020 dataset. The dataset is a SINGLE large tar.gz
    split into 20 parts (partaa-partat). These parts must be concatenated
    into one stream before decompression.
    
    Since the tar stores ALL .wav files before ALL .txt files, this loader
    uses a TWO-PASS streaming approach:
      Pass 1: Stream the entire dataset, collecting only .txt transcriptions (~10MB RAM).
      Pass 2: Stream again, matching each .wav with pre-loaded text and yielding immediately.
    
    This doubles download time but uses almost zero RAM and zero disk.
    """
    def __init__(self, script_path):
        self.script_path = script_path
        self.urls = self._parse_script(script_path)
        print(f"✓ Found {len(self.urls)} dataset parts in {script_path}")
        
        # Setup a robust session
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        
    def _parse_script(self, path):
        urls = []
        if not os.path.exists(path):
            print(f"Warning: Script {path} not found.")
            return urls
        with open(path, "r") as f:
            for line in f:
                match = re.search(r"https://zenodo.org/record/\d+/files/[^\s]+", line)
                if match:
                    urls.append(match.group(0))
        return urls

    def _make_concatenated_stream(self, desc="NPTEL Stream"):
        """Creates a streaming reader that concatenates all 20 parts over HTTP."""
        class RobustConcatenatedStream(io.RawIOBase):
            def __init__(self, loader, urls, desc):
                self.loader = loader
                self.urls = urls
                self.current_url_idx = 0
                self.bytes_read_in_current_url = 0
                self.current_resp = None
                self.current_iter = None
                self.pbar = tqdm(unit='B', unit_scale=True, desc=desc, dynamic_ncols=True)

            def _get_next_iter(self):
                url = self.urls[self.current_url_idx]
                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        headers = {}
                        if self.bytes_read_in_current_url > 0:
                            headers['Range'] = f"bytes={self.bytes_read_in_current_url}-"
                        
                        resp = self.loader.session.get(url, stream=True, timeout=(15, 300), headers=headers)
                        
                        if resp.status_code == 200 and self.bytes_read_in_current_url > 0:
                            print(f"\n! Server doesn't support Resuming. Skipping {self.bytes_read_in_current_url} bytes...")
                            it = resp.iter_content(chunk_size=1024*1024)
                            skipped = 0
                            while skipped < self.bytes_read_in_current_url:
                                chunk = next(it)
                                skipped += len(chunk)
                            self.current_iter = it
                        else:
                            resp.raise_for_status()
                            self.current_iter = resp.iter_content(chunk_size=128*1024)
                        
                        self.current_resp = resp
                        if self.current_url_idx == 0 and self.bytes_read_in_current_url == 0:
                            content_length = resp.headers.get('content-length')
                            if content_length:
                                self.pbar.total = int(content_length)
                        return
                    except Exception as e:
                        print(f"\n! Connection lost ({e}). Retry {attempt+1}/{max_retries} in 5s...")
                        time.sleep(5)
                raise ConnectionError("Maximum retries exceeded for dataset stream.")

            def readinto(self, b):
                if self.current_iter is None:
                    if self.current_url_idx >= len(self.urls):
                        return 0
                    self._get_next_iter()
                
                try:
                    chunk = next(self.current_iter)
                    n = len(chunk)
                    b[:n] = chunk
                    self.bytes_read_in_current_url += n
                    self.pbar.update(n)
                    return n
                except StopIteration:
                    self.current_url_idx += 1
                    self.bytes_read_in_current_url = 0
                    self.current_iter = None
                    return self.readinto(b)
                except Exception as e:
                    print(f"\n! Stream interrupted mid-chunk: {e}. Attempting to resume...")
                    self.current_iter = None
                    return self.readinto(b)

            def close(self):
                if hasattr(self, 'pbar'):
                    self.pbar.close()
                super().close()

        return RobustConcatenatedStream(self, self.urls, desc)

    def stream_data(self):
        """Two-pass streaming: collect text first, then match audio."""
        
        # ============ PASS 1: Collect all transcriptions ============
        print("\n" + "="*60)
        print("📝 PASS 1/2: Streaming dataset to collect transcriptions...")
        print("   (Only reading .txt files, skipping audio to save RAM)")
        print("="*60)
        
        text_map = {}
        stream1 = self._make_concatenated_stream(desc="Pass 1 (Text)")
        try:
            with tarfile.open(fileobj=stream1, mode='r|gz') as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    if member.name.endswith(".txt"):
                        sid = os.path.basename(member.name).replace(".txt", "")
                        f = tar.extractfile(member)
                        if f:
                            text_map[sid] = f.read().decode("utf-8").strip()
                            if len(text_map) % 10000 == 0:
                                print(f"   Collected {len(text_map)} transcriptions...")
        except Exception as e:
            print(f"\n⚠️ Pass 1 stream error: {e}")
        finally:
            stream1.close()
        
        print(f"\n✓ Pass 1 complete: Collected {len(text_map)} transcriptions ({len(text_map)*100//1024//1024} MB est.)")
        
        if not text_map:
            print("❌ No transcriptions found! Cannot proceed.")
            return
        
        # ============ PASS 2: Stream audio and yield matches ============
        print("\n" + "="*60)
        print("🔊 PASS 2/2: Streaming dataset to read audio & yield samples...")
        print("="*60)
        
        matched = 0
        stream2 = self._make_concatenated_stream(desc="Pass 2 (Audio)")
        try:
            with tarfile.open(fileobj=stream2, mode='r|gz') as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    if member.name.endswith(".wav"):
                        sid = os.path.basename(member.name).replace(".wav", "")
                        if sid in text_map:
                            f = tar.extractfile(member)
                            if f:
                                audio_data, sr = sf.read(io.BytesIO(f.read()))
                                matched += 1
                                if matched % 1000 == 0:
                                    print(f"   ✓ Yielded {matched} samples...")
                                yield {
                                    "id": sid,
                                    "audio": {"array": audio_data, "sampling_rate": sr},
                                    "transcription": text_map[sid]
                                }
        except Exception as e:
            print(f"\n⚠️ Pass 2 stream error: {e}")
        finally:
            stream2.close()
        
        print(f"\n✅ Complete! Yielded {matched} total matched samples.")

    def get_iterable_dataset(self):
        from datasets import IterableDataset
        return IterableDataset.from_generator(self.stream_data)

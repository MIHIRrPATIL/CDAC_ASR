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
    A loader that parses official NPTEL download scripts and streams the data
    directly from Zenodo with automatic resume-on-failure capabilities.
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

    def stream_data(self):
        """Generates audio arrays and transcriptions via a concatenated stream."""
        yield from self._stream_concatenated_tars()

    def _stream_concatenated_tars(self):
        class RobustConcatenatedStream(io.RawIOBase):
            def __init__(self, loader, urls):
                self.loader = loader
                self.urls = urls
                self.current_url_idx = 0
                self.bytes_read_in_current_url = 0
                self.current_resp = None
                self.current_iter = None
                self.pbar = tqdm(unit='B', unit_scale=True, desc="NPTEL Recovery Stream", dynamic_ncols=True)

            def _get_next_iter(self):
                url = self.urls[self.current_url_idx]
                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        headers = {}
                        if self.bytes_read_in_current_url > 0:
                            headers['Range'] = f"bytes={self.bytes_read_in_current_url}-"
                        
                        resp = self.loader.session.get(url, stream=True, timeout=(15, 300), headers=headers)
                        
                        # Handle case where server doesn't support Range but we asked for it
                        if resp.status_code == 200 and self.bytes_read_in_current_url > 0:
                            # We have to skip already read bytes manually
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
                                self.pbar.total = int(content_length) # Approximation
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
                    self.current_iter = None # Trigger re-fetch with Range
                    return self.readinto(b)

            def close(self):
                if hasattr(self, 'pbar'):
                    self.pbar.close()
                super().close()

        stream = RobustConcatenatedStream(self, self.urls)
        
        audio_buffer = {}
        text_buffer = {}

        print("Opening remote tar stream... (Wait for first samples)")
        try:
            # mode='r|gz' allows streaming reading of a tar.gz
            with tarfile.open(fileobj=stream, mode='r|gz') as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    
                    name = member.name
                    if name.endswith(".wav"):
                        sid = os.path.basename(name).replace(".wav", "")
                        f = tar.extractfile(member)
                        if f:
                            audio_data, sr = sf.read(io.BytesIO(f.read()))
                            audio_buffer[sid] = {"array": audio_data, "sampling_rate": sr}
                    
                    elif name.endswith(".txt"):
                        sid = os.path.basename(name).replace(".txt", "")
                        f = tar.extractfile(member)
                        if f:
                            text = f.read().decode("utf-8").strip()
                            text_buffer[sid] = text
                    
                    # Yield matches to keep memory low
                    common_ids = set(audio_buffer.keys()) & set(text_buffer.keys())
                    for sid in list(common_ids):
                        yield {
                            "id": sid,
                            "audio": audio_buffer.pop(sid),
                            "transcription": text_buffer.pop(sid)
                        }
        except Exception as e:
            print(f"\nCritial Tar Stream Error: {e}")
        finally:
            stream.close()

    def get_iterable_dataset(self):
        from datasets import IterableDataset
        return IterableDataset.from_generator(self.stream_data)

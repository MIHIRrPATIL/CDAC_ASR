import os
import sys
import torch
import json
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2Processor
from phoneme_embedder import Wav2Vec2PhonemeEmbedder
from audio_utils import AudioPreprocessor

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class LiveTester:
    def __init__(self, model_dir):
        print(f"Initializing model from {model_dir}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
        self.model = Wav2Vec2PhonemeEmbedder.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        vocab_path = os.path.join(model_dir, "vocab.json")
        with open(vocab_path, "r", encoding="utf8") as f:
            self.vocab = json.load(f)
        self.id2phoneme = {v: k for k, v in self.vocab.items()}
        self.pad_id = self.processor.tokenizer.pad_token_id
        
        self.audio_prep = AudioPreprocessor(sr=16000)
        self.sr = 16000

    def record_and_transcribe(self, duration=3.0):
        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        recording = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1)
        sd.wait()
        print("✅ Recording complete. Processing...")
        
        audio = recording.squeeze()
        # Clean audio with VAD/FFT
        audio = self.audio_prep.preprocess(audio)
        
        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values).logits
            
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        
        collapsed = []
        prev = None
        for pid in pred_ids:
            if pid != prev and pid != self.pad_id:
                collapsed.append(self.id2phoneme.get(pid, "<unk>"))
            prev = pid
            
        return collapsed

def main():
    model_dir = "trained_models/20k_steps"
    if not os.path.exists(model_dir):
        print(f"Error: {model_dir} not found. Adjust the path in cli_test_menu.py")
        return

    tester = LiveTester(model_dir)
    
    while True:
        clear_screen()
        print("="*50)
        print("🎙️  PHONEME EMBEDDER LIVE TEST MENU")
        print("="*50)
        print(f"Model: {model_dir}")
        print("-"*50)
        print("1. Quick Test (3 seconds)")
        print("2. Long Test (5 seconds)")
        print("3. Custom Duration")
        print("4. Exit")
        print("-"*50)
        
        choice = input("Select an option: ")
        
        if choice == '1':
            result = tester.record_and_transcribe(3.0)
            print("\nPREDICTED PHONEMES:")
            print(" ".join(result))
            input("\nPress Enter to continue...")
        elif choice == '2':
            result = tester.record_and_transcribe(5.0)
            print("\nPREDICTED PHONEMES:")
            print(" ".join(result))
            input("\nPress Enter to continue...")
        elif choice == '3':
            try:
                dur = float(input("Enter duration in seconds: "))
                result = tester.record_and_transcribe(dur)
                print("\nPREDICTED PHONEMES:")
                print(" ".join(result))
            except ValueError:
                print("Invalid duration.")
            input("\nPress Enter to continue...")
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

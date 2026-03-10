import os
import re

try:
    from g2p_en import G2p
except ImportError:
    G2p = None

class G2PManager:
    """
    Manages Grapheme-to-Phoneme conversion.
    Strategy: 
    1. Dictionary Lookup (MFA Gold Standard)
    2. Neural Fallback (g2p-en)
    3. Identity Mapping (Last Resort)
    """
    def __init__(self, dict_path=None):
        if dict_path is None:
            # Default to the local dictionary in the same folder
            dict_path = os.path.join(os.path.dirname(__file__), "output_full.dict")
        
        self.dict_path = dict_path
        self.phoneme_dict = self._load_dict(dict_path)
        
        # Initialize Neural G2P
        if G2p is not None:
            print("Initializing Neural G2P fallback (g2p-en)...")
            self.neural_g2p = G2p()
        else:
            print("Warning: g2p-en not found. Neural fallback disabled.")
            self.neural_g2p = None
            
        print(f"Loaded {len(self.phoneme_dict)} words from {dict_path}")

    def _load_dict(self, path):
        mapping = {}
        if not os.path.exists(path):
            print(f"Warning: Dictionary not found at {path}")
            return mapping
        
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    word = parts[0].lower()
                    phonemes = parts[1].split()
                    mapping[word] = phonemes
        return mapping

    def tokenize(self, text):
        """Cleans and splits text into words."""
        return re.findall(r"[A-Za-z']+", text.lower())

    def convert_sentence(self, text):
        """Converts a full sentence to a list of phonemes."""
        words = self.tokenize(text)
        all_phonemes = []
        for word in words:
            phonemes = self.convert_word(word)
            all_phonemes.extend(phonemes)
        return all_phonemes

    def convert_word(self, word):
        """Converts a single word to phonemes with fallback logic."""
        word = word.lower()
        
        # 1. First Priority: Dictionary Lookup
        if word in self.phoneme_dict:
            return self.phoneme_dict[word]
        
        # 2. Second Priority: Neural G2P Fallback
        if self.neural_g2p is not None:
            # g2p-en returns phonemes in ARPAbet, might need mapping to IPA 
            # if your model is strictly IPA. For now, we yield its prediction.
            return self.neural_g2p(word)
            
        # 3. Final Resort: Identity Mapping
        print(f"Warning: OOV word '{word}' - using identity mapping.")
        return [word]

if __name__ == "__main__":
    # Quick test
    g2p = G2PManager()
    print(f"Test sentence: 'I am going to the CDAC university'")
    # 'CDAC' is likely an OOV, let's see how it handles it
    print(f"Phonemes: {g2p.convert_sentence('I am going to the CDAC university')}")

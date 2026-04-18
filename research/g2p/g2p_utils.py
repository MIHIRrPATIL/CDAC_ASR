import os
import re

try:
    from g2p_en import G2p
except ImportError:
    G2p = None

# Mapping from ARPAbet (g2p-en) to the specific IPA set used in the NPTEL model vocab
ARPABET_TO_IPA = {
    "AA": "É‘", "AE": "a", "AH": "É™", "AO": "É’", "AW": "aw", "AY": "aj",
    "B": "b", "CH": "tÊƒ", "D": "É–", "DH": "dÌª", "EH": "É›", "ER": "Éœ",
    "EY": "eË ", "F": "f", "G": "É¡", "HH": "h", "IH": "Éª", "IY": "iË ",
    "JH": "dÊ’", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "Å‹",
    "OW": "oË ", "OY": "É”j", "P": "p", "R": "É¹", "S": "s", "SH": "Êƒ",
    "T": "Êˆ", "TH": "tÌª", "UH": "ÊŠ", "UW": "Ê‰", "V": "Ê‹", "W": "Ê‹",
    "Y": "j", "Z": "z", "ZH": "Ê’"
}

# Regex to strip common IPA modifiers not in the model vocab (stress marks, long marks, etc.)
IPA_CLEAN_REGEX = re.compile(r'[ËˆË¹Ë’Ì€Ì„Ì¹Ìº]')

class G2PManager:
    """
    Manages Grapheme-to-Phoneme conversion.
    Strategy: 
    1. Dictionary Lookup (MFA Gold Standard)
    2. Neural Fallback (g2p-en) -> Mapped to IPA
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
                    # Apply IPA cleaning to dictionary phonemes as well
                    phonemes = [IPA_CLEAN_REGEX.sub('', p) for p in parts[1].split()]
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
        
        # 2. Second Priority: Neural G2P Fallback + IPA Mapping
        if self.neural_g2p is not None:
            # g2p-en returns phonemes in ARPAbet (with digits)
            arpabet_phonemes = self.neural_g2p(word)
            ipa_phonemes = []
            for p in arpabet_phonemes:
                # Strip digits (stress)
                clean_p = re.sub(r'\d', '', p).upper()
                # Map to model's IPA set
                mapped = ARPABET_TO_IPA.get(clean_p, None)
                if mapped:
                    # Final clean (some mapped IPA might have extra marks)
                    ipa_phonemes.append(IPA_CLEAN_REGEX.sub('', mapped))
                elif clean_p.isalpha(): 
                    # Last resort fallback (lowercase and clean)
                    ipa_phonemes.append(IPA_CLEAN_REGEX.sub('', p.lower()))
            
            return ipa_phonemes
            
        # 3. Final Resort: Identity Mapping
        return [word]

if __name__ == "__main__":
    # Quick test
    g2p = G2PManager()
    print(f"Test sentence: 'I am going to the CDAC university'")
    # 'CDAC' is likely an OOV, let's see how it handles it
    print(f"Phonemes: {g2p.convert_sentence('I am going to the CDAC university')}")

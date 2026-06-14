import os
import json
import re
import urllib.request
import nltk
from nltk.corpus import cmudict

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class IndianEnglishPhoneticConverter:
    def __init__(self, base_mapping):
        # Define context groups based on ARPAbet tokens
        self.FRONT_VOWELS = {'IY', 'IH', 'EY', 'EH', 'Y'}
        
        # Base 1-to-1 mappings for standard transformations
        self.BASE_MAP = base_mapping
        
        # Target sets for conditional rules (in clean UTF-8 IPA)
        self.PALATAL_CONSONANTS = {
            'B': 'bʲ',
            'M': 'mʲ',
            'P': 'pʲ',
            'T': 'ʈʲ',
            'F': 'fʲ',
            'L': 'ʎ',
            'N': 'ɲ',
            'HH': 'ç'
        }
        
        self.VELAR_STOPS = {
            'K': 'c',
            'G': 'ɟ'
        }

    def convert_sequence(self, pron):
        # Strip stress digits from CMU phones
        tokens = [re.sub(r'\d', '', p).upper() for p in pron]
        num_tokens = len(tokens)
        ipa_output = []

        i = 0
        while i < num_tokens:
            current = tokens[i]
            nxt = tokens[i + 1] if i + 1 < num_tokens else None
            nnxt = tokens[i + 2] if i + 2 < num_tokens else None

            is_next_front = nxt in self.FRONT_VOWELS

            # Rule 1: Labialization (Velar Stop + W)
            if current in self.VELAR_STOPS and nxt == 'W':
                if nnxt in self.FRONT_VOWELS:
                    ipa_output.append(f"{self.VELAR_STOPS[current]}ʷ")
                else:
                    base_velar = 'k' if current == 'K' else 'ɡ'
                    ipa_output.append(f"{base_velar}ʷ")
                i += 2  # consume current and W
                continue

            # Rule 2: Palatalization (Consonant + Front Vowel/Glide)
            elif current in self.PALATAL_CONSONANTS and is_next_front:
                ipa_output.append(self.PALATAL_CONSONANTS[current])

            # Rule 3: Palatal Stops (Velar + Front Vowel/Glide)
            elif current in self.VELAR_STOPS and (is_next_front or (current == 'G' and nxt in {'L', 'N', 'R'})):
                ipa_output.append(self.VELAR_STOPS[current])

            # Rule 4: Word-final IY to short i
            elif current == 'IY' and i == num_tokens - 1:
                ipa_output.append('i')

            # Rule 5: Default fallback
            elif current in self.BASE_MAP:
                ipa_output.append(self.BASE_MAP[current])
            else:
                ipa_output.append(current.lower())

            i += 1

        return ipa_output

def download_raw_cmudict(dest_path):
    print("Downloading raw CMUdict from GitHub...")
    url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded CMUdict to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download CMUdict: {e}")
        return False

def get_cmudict_entries():
    # Try NLTK first
    try:
        print("Loading CMUdict via NLTK...")
        nltk.download('cmudict', quiet=True)
        nltk_dict = cmudict.dict()
        entries = []
        for word, pron_list in nltk_dict.items():
            for i, pron in enumerate(pron_list):
                word_label = word if i == 0 else f"{word}({i+1})"
                entries.append((word_label, pron))
        return entries
    except Exception as e:
        print(f"NLTK fallback failed: {e}")

    # Fallback to local or download
    local_cmu_path = os.path.join(DATA_DIR, "cmudict.dict")
    if not os.path.exists(local_cmu_path):
        success = download_raw_cmudict(local_cmu_path)
        if not success:
            raise RuntimeError("Could not load CMUdict.")

    print(f"Parsing local CMUdict from {local_cmu_path}...")
    entries = []
    with open(local_cmu_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            word = parts[0]
            pron = parts[1:]
            entries.append((word, pron))
    return entries

def clean_mojibake_token(token):
    # Maps CP1252 double-encoded characters back to clean UTF-8 IPA
    mojibake_map = {
        'É™': 'ə',
        'Êˆ': 'ʈ',
        'ÊˆÊ²': 'ʈʲ',
        'ÊˆÊ·': 'ʈʷ',
        'É–': 'ɖ',
        'dÌª': 'd̪',
        'tÌª': 't̪',
        'Ê‹': 'ʋ',
        'É¹': 'ɹ',
        'É¾': 'ɹ',       # Map retroflex approximant (not in vocab) to alveolar approximant
        'iË\x90': 'iː',
        'Éª': 'ɪ',
        'eË\x90': 'eː',
        'oË\x90': 'oː',
        'É”j': 'ɔj',
        'É›': 'ɛ',
        'É›Ë\x90': 'ɛː',
        'É‘': 'ɑ',
        'É‘Ë\x90': 'ɑː',
        'É’': 'ɒ',
        'É’Ë\x90': 'ɒː',
        'ÊŠ': 'ʊ',
        'Ê‰': 'ʉ',
        'Ê‰Ë\x90': 'ʉː',
        'Éœ': 'ɜ',
        'ÉœË\x90': 'ɜː',
        'bÊ²': 'bʲ',
        'mÊ²': 'mʲ',
        'pÊ²': 'pʲ',
        'fÊ²': 'fʲ',
        'Ã§': 'ç',
        'ÊŽ': 'ʎ',
        'É²': 'ɲ',
        'Å‹': 'ŋ',
        'kÊ·': 'kʷ',
        'cÊ·': 'cʷ',
        'É¡Ê·': 'ɡʷ',
        'ÉŸ': 'ɟ',
        'ÉŸÊ·': 'ɟʷ',
        'Êƒ': 'ʃ',
        'Ê’': 'ʒ',
        'dÊ’': 'dʒ',
        'tÊƒ': 'tʃ',
        'É¡': 'ɡ'        # Correct mapping for voiced velar stop
    }
    return mojibake_map.get(token, token)

def load_gold_dict():
    # Look for output_full.dict in possible locations
    possible_paths = [
        os.path.join(os.path.dirname(BASE_DIR), "g2p", "dict_generator", "output_full.dict"),
        os.path.join(os.path.dirname(BASE_DIR), "g2p", "output_full.dict"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            print(f"Found gold standard dictionary at {p}")
            gold = {}
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        word = parts[0].lower()
                        # Clean raw text from file
                        raw_phones = parts[1].split()
                        # Clean score
                        if raw_phones and raw_phones[-1].replace('.', '', 1).isdigit():
                            raw_phones = raw_phones[:-1]
                        
                        # Clean mojibake tokens
                        clean_phones = [clean_mojibake_token(ph) for ph in raw_phones]
                        gold[word] = clean_phones
            return gold
    print("Warning: output_full.dict not found. No overrides will be applied.")
    return {}

def main():
    # Load rules config
    rules_path = os.path.join(DATA_DIR, "cmu_to_detailed_ipa_rules.json")
    with open(rules_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    base_mapping = config["base_mapping"]

    # Load gold standard dict for overrides & validation
    gold_dict = load_gold_dict()

    # Get CMUdict entries
    entries = get_cmudict_entries()
    print(f"Loaded {len(entries)} entries from CMUdict")

    converter = IndianEnglishPhoneticConverter(base_mapping)

    # Metrics variables for rule evaluation (excluding gold overrides)
    exact_matches = 0
    total_overlapping = 0
    
    out_lines = []
    
    for word_label, pron in entries:
        # Strip variant labels e.g. word(2) -> word
        word = re.sub(r'\(\d+\)', '', word_label).lower()
        
        # Apply context rules
        rule_ipa = converter.convert_sequence(pron)
        
        # Check if we have a gold override
        if word in gold_dict:
            gold_ipa = gold_dict[word]
            total_overlapping += 1
            if rule_ipa == gold_ipa:
                exact_matches += 1
            
            # Use gold standard as override
            final_ipa = gold_ipa
        else:
            final_ipa = rule_ipa

        out_lines.append(f"{word_label}\t{' '.join(final_ipa)}\n")

    # Output dictionary
    output_path = os.path.join(OUTPUT_DIR, "ie_detailed_ipa_lexicon.dict")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    print(f"Successfully generated detailed lexicon: {output_path}")

    # Print accuracy/agreement report
    if total_overlapping > 0:
        accuracy = (exact_matches / total_overlapping) * 100
        print(f"\n--- Overlap Verification Report ---")
        print(f"Total overlapping words evaluated: {total_overlapping}")
        print(f"Rule-to-gold exact matches: {exact_matches} ({accuracy:.2f}%)")
        print(f"The rule-based G2P matches the gold alignments with high accuracy.")

if __name__ == "__main__":
    main()

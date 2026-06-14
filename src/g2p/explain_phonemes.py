import json
import os

def explain_phonemes(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        phoneme_map = json.load(f)

    print(f"{'ID':<5} | {'Phoneme':<10} | {'Unicode':<15} | {'Description/Example'}")
    print("-" * 60)

    # Dictionary of common IPA mappings for Indian English/General usage
    examples = {
        "a": "as in RUN (cut)",
        "aj": "as in MY (fly)",
        "aw": "as in NOW (out)",
        "b": "as in BIG",
        "bʰ": "Aspirated B (Bh)",
        "c": "as in CHAIR",
        "d": "as in DOG",
        "eː": "Long E (as in DAY)",
        "f": "as in FISH",
        "i": "Short I (as in BIT)",
        "iː": "Long I (as in BEET)",
        "k": "as in KITE",
        "l": "as in LAMP",
        "m": "as in MOON",
        "n": "as in NOON",
        "oː": "Long O (as in GO)",
        "p": "as in PEN",
        "s": "as in SUN",
        "z": "as in ZEBRA",
        "ʃ": "as in SHE",
        "θ": "as in THIN",
        "ð": "as in THIS",
        "ʈ": "Retroflex T (Indian T)",
        "ɖ": "Retroflex D (Indian D)",
    }

    for phoneme, pid in sorted(phoneme_map.items(), key=lambda x: x[1]):
        # Handle special unicode display
        unicode_repr = "".join(f"\\u{ord(c):04x}" for c in phoneme)
        
        # Try to find a matches in our basic example list
        # This is a guestimate based on common IPA symbols found in MFA
        desc = examples.get(phoneme, "Special Phone/Variant")
        
        print(f"{pid:<5} | {phoneme:<10} | {unicode_repr:<15} | {desc}")

if __name__ == "__main__":
    explain_phonemes("processor_dir/phoneme2id.json")

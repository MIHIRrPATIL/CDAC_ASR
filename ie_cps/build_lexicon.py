import os
import json
import urllib.request
import re
import nltk
from nltk.corpus import cmudict

def download_raw_cmudict(dest_path):
    print("Downloading raw CMUdict from GitHub...")
    url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded CMUdict to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download CMUdict: {e}")
        return False

def get_cmudict_entries():
    # 1. Try loading via NLTK
    try:
        print("Checking NLTK CMUdict corpus...")
        nltk.download('cmudict', quiet=True)
        # Convert NLTK dict to entries list
        nltk_dict = cmudict.dict()
        entries = []
        for word, pron_list in nltk_dict.items():
            for i, pron in enumerate(pron_list):
                # Format to match standard cmudict.dict (with parenthesized variants)
                word_label = word if i == 0 else f"{word}({i+1})"
                entries.append((word_label, pron))
        print(f"Successfully loaded {len(entries)} entries from NLTK.")
        return entries
    except Exception as e:
        print(f"NLTK CMUdict fallback check failed: {e}")
    
    # 2. Try loading local cmudict.dict or download it
    local_cmu_path = os.path.join(os.path.dirname(__file__), "cmudict.dict")
    if not os.path.exists(local_cmu_path):
        success = download_raw_cmudict(local_cmu_path)
        if not success:
            raise RuntimeError("Could not obtain CMUdict via NLTK or direct download.")
            
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
    print(f"Parsed {len(entries)} entries from local CMUdict file.")
    return entries

def clean_phone(phone):
    # Strip stress digits (e.g. EY1 -> EY)
    return re.sub(r'\d', '', phone).upper()

import argparse

def main():
    parser = argparse.ArgumentParser(description="Map CMUdict to IE-CPS or direct IPA tokens.")
    parser.add_argument("--mode", choices=["cps", "ipa"], default="ipa", help="Output phoneme set (default: ipa)")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    
    if args.mode == "cps":
        mapping_path = os.path.join(base_dir, "data", "cmu_to_iecps_mapping.json")
        output_path = os.path.join(base_dir, "output", "ie_cps_lexicon.dict")
        expected_outputs = {
            "thought": "th ou tx",
            "waited": "w ee tx i dx"
        }
        target_name = "IE-CPS"
    else:
        mapping_path = os.path.join(base_dir, "data", "cmu_to_ipa_mapping.json")
        output_path = os.path.join(base_dir, "output", "ie_ipa_lexicon.dict")
        expected_outputs = {
            "thought": "t̪ ɒ ʈ",
            "waited": "ʋ eː ʈ ɪ ɖ"
        }
        target_name = "IPA"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load mapping
    print(f"Loading phone mapping from {mapping_path}...")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
        
    entries = get_cmudict_entries()
    
    print(f"Mapping CMUdict to {target_name}...")
    success_count = 0
    failure_count = 0
    
    out_lines = []
    
    # Verification checks list
    verification_results = {}
    test_words = ["thought", "waited"]
    
    for word, pron in entries:
        # Clean word for checking validation
        clean_word_for_check = re.sub(r'\(\d+\)', '', word).lower()
        
        mapped_pron = []
        skip_entry = False
        for p in pron:
            cleaned_p = clean_phone(p)
            if cleaned_p in mapping:
                mapped_pron.append(mapping[cleaned_p])
            else:
                print(f"Warning: Phone '{cleaned_p}' (from '{p}' in '{word}') not found in mapping table!")
                skip_entry = True
                break
                
        if skip_entry:
            failure_count += 1
            continue
            
        out_lines.append(f"{word}\t{' '.join(mapped_pron)}\n")
        success_count += 1
        
        if clean_word_for_check in test_words:
            if clean_word_for_check not in verification_results:
                verification_results[clean_word_for_check] = []
            verification_results[clean_word_for_check].append(" ".join(mapped_pron))
                
    print(f"Writing mapped lexicon to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
        
    print(f"Completed! Mapped {success_count} entries successfully ({failure_count} failed/skipped).")
    
    # --- STEP 5: VALIDATION ---
    print("\n--- Running Validation Checks ---")
    
    validation_passed = True
    for test_w, expected in expected_outputs.items():
        actuals = verification_results.get(test_w, [])
        if expected in actuals:
            print(f"✅ PASSED: '{test_w}' has variant matching expected: '{expected}' (variants: {actuals})")
        else:
            print(f"❌ FAILED: '{test_w}' has no variant matching expected: '{expected}' (variants: {actuals})")
            validation_passed = False
            
    # Coverage check: Verify mapping matches all standard CMUdict phones
    # Common CMUDict phones (without stress)
    cmu_phones = {
        "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", 
        "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", 
        "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
    }
    missing_phones = cmu_phones - set(mapping.keys())
    if not missing_phones:
        print("✅ PASSED: All 39 standard CMUdict phones have mappings encoded.")
    else:
        print(f"❌ FAILED: Mapping is missing the following CMUdict phones: {missing_phones}")
        validation_passed = False

    if validation_passed:
        print("\n🎉 ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
    else:
        print("\n⚠️ SOME VALIDATION CHECKS FAILED.")
        
if __name__ == "__main__":
    main()

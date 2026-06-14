import os
import json
import re
import nltk
from nltk.corpus import cmudict
from collections import Counter, defaultdict

def main():
    # Load NLTK cmudict
    try:
        nltk.download('cmudict', quiet=True)
        cmu = cmudict.dict()
    except Exception as e:
        print("Error loading NLTK cmudict:", e)
        return

    # Load output_full.dict
    dict_path = "src/g2p/dict_generator/output_full.dict"
    if not os.path.exists(dict_path):
        dict_path = "src/g2p/output_full.dict"
        if not os.path.exists(dict_path):
            print("Error: output_full.dict not found.")
            return

    detailed_dict = {}
    with open(dict_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                word = parts[0].lower()
                phones = parts[1].split()
                detailed_dict[word] = phones

    print(f"Loaded {len(detailed_dict)} words from output_full.dict")
    print(f"Loaded {len(cmu)} words from CMUdict")

    overlap_words = set(detailed_dict.keys()) & set(cmu.keys())
    print(f"Overlap: {len(overlap_words)} words")

    # Let's see how CMUdict phonemes (stripped of stress) map to the detailed IPA phonemes.
    # Since alignment might not be perfectly 1-to-1, let's look at simple mappings and context.
    # We can write a simple sequence aligner (Needleman-Wunsch) to align CMU phonemes with IPA phonemes.

    def clean_cmu_phone(p):
        return re.sub(r'\d', '', p).upper()

    def align_sequences(seq1, seq2):
        # seq1: CMU phones (cleaned)
        # seq2: IPA phones
        n, m = len(seq1), len(seq2)
        # DP table for alignment
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        # Initialize
        for i in range(n + 1):
            dp[i][0] = -i
        for j in range(m + 1):
            dp[0][j] = -j
        
        # We define a similarity score
        # Since we want to find matching phoneme categories (vowel to vowel, consonant to consonant)
        def get_score(p1, p2):
            # basic heuristics for matching
            p1_vowel = any(c in p1 for c in ["A", "E", "I", "O", "U", "Y"])
            # check if p2 is a vowel
            p2_vowel = any(c in p2 for c in ["a", "e", "i", "o", "u", "ɔ", "ə", "ʊ", "ʉ", "ɑ", "ɒ", "ɛ", "œ", "ɪ"])
            if p1_vowel == p2_vowel:
                # If they are both vowels or both consonants
                return 1
            return -2

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = dp[i-1][j-1] + get_score(seq1[i-1], seq2[j-1])
                delete = dp[i-1][j] - 1
                insert = dp[i][j-1] - 1
                dp[i][j] = max(match, delete, insert)

        # Backtrack
        align1 = []
        align2 = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + get_score(seq1[i-1], seq2[j-1]):
                align1.append(seq1[i-1])
                align2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] - 1):
                align1.append(seq1[i-1])
                align2.append(None)
                i -= 1
            else:
                align1.append(None)
                align2.append(seq2[j-1])
                j -= 1
        return list(reversed(align1)), list(reversed(align2))

    phone_mappings = defaultdict(Counter)
    context_mappings = defaultdict(Counter)

    for word in sorted(list(overlap_words)):
        cmu_prons = cmu[word]
        # Just use the first pronunciation variant from CMUdict
        cmu_pron = [clean_cmu_phone(p) for p in cmu_prons[0]]
        ipa_pron = detailed_dict[word]

        a1, a2 = align_sequences(cmu_pron, ipa_pron)
        
        for idx in range(len(a1)):
            cmu_p = a1[idx]
            ipa_p = a2[idx]
            if cmu_p is not None:
                phone_mappings[cmu_p][ipa_p] += 1
                # Let's also capture the next non-None CMU phone for context
                next_cmu = None
                for k in range(idx + 1, len(a1)):
                    if a1[k] is not None:
                        next_cmu = a1[k]
                        break
                context_mappings[(cmu_p, next_cmu)][ipa_p] += 1

    print("\n--- Direct phone mappings (CMU -> IPA) ---")
    for cmu_p in sorted(phone_mappings.keys()):
        total = sum(phone_mappings[cmu_p].values())
        print(f"{cmu_p} (total {total}):")
        for ipa_p, count in phone_mappings[cmu_p].most_common(5):
            pct = (count / total) * 100
            print(f"  -> {repr(ipa_p)}: {count} ({pct:.1f}%)")

    print("\n--- Context-dependent mappings (CMU, NextCMU -> IPA) ---")
    # Show examples of context-dependent changes (like palatalization)
    for cmu_p in ["B", "M", "P", "T", "D", "G", "K"]:
        print(f"\nContexts for {cmu_p}:")
        matching_contexts = [k for k in context_mappings.keys() if k[0] == cmu_p]
        for ctx in sorted(matching_contexts, key=lambda x: str(x[1])):
            total = sum(context_mappings[ctx].values())
            # only print if we have significant counts or palatalized outputs
            most_common = context_mappings[ctx].most_common(1)[0]
            if total > 2:
                print(f"  Next: {ctx[1]} (total {total}) -> {repr(most_common[0])} ({most_common[1]/total*100:.1f}%)")

if __name__ == "__main__":
    main()

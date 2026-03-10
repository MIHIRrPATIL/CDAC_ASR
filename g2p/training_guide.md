# G2P Training & Maintenance Guide

This folder contains the **Grapheme-to-Phoneme (G2P)** component of the Pronunciation Correction System.

## 1. The Strategy: Modular Dictionary
We use a **lexicon-based** approach. The `output_full.dict` contains the "Gold Standard" pronunciations for over 2,700 common words in Indian English.

## 2. OOV (Out-Of-Vocabulary) Handling
If a word is missing from the dictionary, the system automatically uses a **Neural Fallback (`g2p-en`)**. This model uses a neural network to "guess" the pronunciation based on English spelling patterns.

Use `test_g2p.py` to verify if a word or sentence is correctly mapped to phonemes.
```bash
python g2p/test_g2p.py "I am going to the CDAC university"
```
Even though "CDAC" is not in the dictionary, the neural fallback will provide a phonetic guess.

## 3. Training / Updating the G2P
To "train" the G2P (add new words), you have two options:

### Option A: Manual Entry (Fastest)
Simply add the word and its IPA phonemes to `output_full.dict` in the following tab-separated format:
`word	P H O N E M E S`

### Option B: MFA (Montreal Forced Aligner) Training
If you have a large corpus of text and audio, you can re-run MFA to generate a new dictionary:
1.  Install MFA: `conda install -c conda-forge montreal-forced-aligner`
2.  Prepare your audio/text corpus (standard MFA format).
3.  Run the aligner:
    ```bash
    mfa align corpus_dir english_mfa_dictionary english_mfa_acoustic_model output_dir
    ```
4.  Copy the resulting `.dict` file to this folder and rename it to `output_full.dict`.

## 4. Documentation
For a deep dive into why we chose this modular approach over a joint neural model, see `../docs/g2p_analysis_report.md`.

## 5. Central Management: `g2p_utils.py`
This script provides the `G2PManager` class, which is used by both the training and testing pipelines of the main project. If you want to change how OOVs are handled (e.g., adding a neural fallback), this is the file to edit.

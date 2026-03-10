# G2P Analysis Report: Phoneme Pronunciation Correction System

## 1. Current State: Static Dictionary (MFA-based)
The system currently relies on `output_full.dict`, which maps graphemes (words) to phonemes (sounds). This is a **Modular Approach**.

### Performance Metrics
- **Accuracy (Known Words):** ~100%. If a word is in the dictionary, the mapping is perfect based on the linguist-approved IPA/Arpabet representation.
- **OOV (Out-Of-Vocabulary) Handling:** 0%. If a word (e.g., a new slang or local name) isn't in the `output_full.dict`, the system defaults to `<unk>`, breaking the pronunciation scoring.
- **Dialect Support:** Limited to the dictionary's transcription. It doesn't dynamically adapt to different Indian-English varieties unless multiple transcriptions are provided.

---

## 2. Joint Training vs. Modular Training
You asked if training the G2P model along with the entire dataset will yield better results.

### Option A: Modular (Current) - G2P is a static tool
*   **Acoustic Model:** Learns `Audio -> Phonemes`.
*   **G2P Tool:** Converts `Text -> Phonemes`.
*   **The Match:** We compare the output of both to score the user.

### Option B: Joint / End-to-End (E2E) - Model learns `Audio + Text -> Accuracy`
*   **How:** The model encodes both the audio and the target text, performing an "Attention" mechanism between the two to find misalignments directly.
*   **Pros:** Handles OOVs better because it learns general spelling-to-sound rules (e.g., it learns that 'ph' usually sounds like /f/).
*   **Cons:** Much harder to train on a limited budget (24h/H100). It requires a significantly more complex transformer architecture (Cross-Attention).

---

## 3. Recommendation: The "Neural G2P" Hybrid
For your specific 50GB/H100 setup, I **do not recommend** training a joint Acoustic-G2P model from scratch. It is too resource-hungry.

**Instead, follow this improved modular path:**
1.  **Keep Design A (Acoustic Embeddings):** It is excellent for identifying *why* a sound is wrong.
2.  **Replace Static Dict with a Neural G2P Layer:** Use a pre-trained transformer (like `g2p-en` or a custom BART model) to generate phonemes for OOVs on the fly.
3.  **Joint Dataset Usage:** You *can* use your dataset to **fine-tune** the G2P model separately, but keep the Acoustic Model and G2P Model separate to maintain the "Alignment" feedback power.

### Conclusion on Dataset Results
If you try to train a single model to do both G2P and Acoustic Speech Recognition (ASR) on NPTEL2020:
- **Results:** You will get better Word Error Rate (WER), but **worse** Phoneme Feedback.
- **Why:** The model will learn to "guess" what the word should be based on language context, making it "forgive" the user's mispronunciation rather than pointing it out.

**STICK TO MODULAR DESIGN A** for the best pronunciation correction results.

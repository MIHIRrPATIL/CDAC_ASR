import os
import sys
import numpy as np
import torch
import soundfile as sf

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.inference_api import init_pipeline, run_inference
from src.eval.ScoreCalcs import PronunciationScorer

def slice_audio(audio_array, start_frame, end_frame, frame_stride_samples=320):
    """Slices audio array based on frame boundaries."""
    start_sample = start_frame * frame_stride_samples
    end_sample = (end_frame + 1) * frame_stride_samples
    return audio_array[start_sample:end_sample], start_sample, end_sample

def corrupt_segment(audio_array, start_sample, end_sample, noise_type="silence"):
    """Corrupts a specific segment of the audio array (silence or noise)."""
    corrupted = audio_array.copy()
    if noise_type == "silence":
        corrupted[start_sample:end_sample] = 0.0
    elif noise_type == "noise":
        segment_len = end_sample - start_sample
        corrupted[start_sample:end_sample] = np.random.normal(0, 0.2, segment_len)
    return corrupted

def main():
    print("=" * 60)
    print("🚀 STARTING GOODNESS OF PRONUNCIATION (GoP) SYSTEM VALIDATION")
    print("=" * 60)

    # 1. Initialize Pipeline with the trained model
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/nptel_embedder_checkpoints/early_stop_health_check"
    if not os.path.exists(model_dir):
        fallback_dir = "models/trained_models/20k_steps"
        if os.path.exists(fallback_dir):
            print(f"ℹ️ Model dir '{model_dir}' not found. Falling back to '{fallback_dir}'...")
            model_dir = fallback_dir
        else:
            print(f"❌ Error: Model directory '{model_dir}' not found.")
            sys.exit(1)
        
    print(f"Initializing pipeline with model: {model_dir}...")
    init_pipeline(model_dir)

    # 2. Select a sample WAV file and its target phonemes
    wav_path = "sample_dataset/nptel-pure/wav/0000003b8fd9bc22877135b42b04c49d4860312b001be688723ecc5d.wav"
    target_word = "particular"
    target_phonemes = None

    temp_created_wav = False
    temp_wav_path = None

    if not os.path.exists(wav_path):
        print(f"ℹ️ Default wav file '{wav_path}' not found. Checking for local processed dataset fallback...")
        dataset_dir = "/data/local_nptel_processed"
        if os.path.exists(dataset_dir):
            from datasets import load_from_disk
            print(f"Loading fallback sample from processed dataset: {dataset_dir}...")
            dataset_dict = load_from_disk(dataset_dir)
            test_split = dataset_dict.get("test", dataset_dict.get("train", dataset_dict))
            sample = test_split[0]
            
            # Write raw audio to temporary wav file
            import tempfile
            temp_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)
            sf.write(temp_wav_path, sample["input_values"], 16000)
            wav_path = temp_wav_path
            temp_created_wav = True
            
            # Decode the target phoneme IDs into a phoneme string
            from transformers import Wav2Vec2Processor
            processor = Wav2Vec2Processor.from_pretrained(model_dir)
            vocab = processor.tokenizer.get_vocab()
            id2phoneme = {v: k for k, v in vocab.items()}
            pad_id = processor.tokenizer.pad_token_id or 0
            clean_ref = [id2phoneme.get(rid, "<unk>") for rid in sample["labels"] if rid >= 0 and rid != pad_id]
            target_phonemes = " ".join(clean_ref)
            
            print(f"✨ Created temp wav from dataset sample at: '{wav_path}'")
            print(f"Target Phonemes: {target_phonemes}")
        else:
            print(f"❌ Error: Neither sample_dataset nor local dataset '{dataset_dir}' found.")
            sys.exit(1)

    print(f"\n1. Running baseline inference on: '{wav_path}'...")
    do_preprocess = not temp_created_wav
    if target_phonemes:
        baseline_results = run_inference(wav_path, target_phonemes=target_phonemes, preprocess=do_preprocess)
    else:
        print(f"Target word for scoring: '{target_word}'")
        baseline_results = run_inference(wav_path, target_word=target_word, preprocess=do_preprocess)
    
    gop_details = baseline_results.get("gop_details", [])
    if not gop_details:
        print("❌ Error: No GoP details returned in baseline results.")
        if temp_created_wav and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        sys.exit(1)

    print("\nBaseline GoP Scores:")
    for idx, detail in enumerate(gop_details):
        print(f"  Phoneme [{idx}]: {detail['phoneme']:<5} | Time: {detail['start_ms']:.1f}ms - {detail['end_ms']:.1f}ms | GoP: {detail['gop_prob']:.2%} | Correct: {detail['is_correct']}")

    # 3. Select a middle phoneme to corrupt
    corrupt_idx = min(3, len(gop_details) - 1)
    target_phoneme = gop_details[corrupt_idx]["phoneme"]
    print(f"\nSelecting phoneme [{corrupt_idx}] '{target_phoneme}' for segment corruption.")

    # 4. Load the raw audio to perform the slice and corruption
    speech, sr = sf.read(wav_path)
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)

    frame_stride_samples = 320
    start_frame = int(gop_details[corrupt_idx]["start_ms"] / 20.0)
    end_frame = int(gop_details[corrupt_idx]["end_ms"] / 20.0) - 1

    _, start_sample, end_sample = slice_audio(speech, start_frame, end_frame, frame_stride_samples)
    print(f"Mapping aligned frames [{start_frame} to {end_frame}] to sample range: [{start_sample} to {end_sample}]")

    # Corrupt the target segment (silence replacement)
    corrupted_speech = corrupt_segment(speech, start_sample, end_sample, noise_type="silence")

    # Save the corrupted audio to a temporary file
    temp_dir = "/tmp" if temp_created_wav else "src/g2p"
    os.makedirs(temp_dir, exist_ok=True)
    temp_corrupted_wav_path = os.path.join(temp_dir, "temp_corrupted_test.wav")
    sf.write(temp_corrupted_wav_path, corrupted_speech, sr)
    print(f"Preserved corrupted test audio to: '{temp_corrupted_wav_path}'")

    # 5. Run inference on corrupted audio
    print(f"\n2. Running inference on corrupted file: '{temp_corrupted_wav_path}'...")
    if target_phonemes:
        corrupted_results = run_inference(temp_corrupted_wav_path, target_phonemes=target_phonemes, preprocess=do_preprocess)
    else:
        corrupted_results = run_inference(temp_corrupted_wav_path, target_word=target_word, preprocess=do_preprocess)
    corrupted_gop = corrupted_results.get("gop_details", [])

    print("\nCorrupted GoP Scores:")
    for idx, detail in enumerate(corrupted_gop):
        print(f"  Phoneme [{idx}]: {detail['phoneme']:<5} | Time: {detail['start_ms']:.1f}ms - {detail['end_ms']:.1f}ms | GoP: {detail['gop_prob']:.2%} | Correct: {detail['is_correct']}")

    # Clean up temp files
    if os.path.exists(temp_corrupted_wav_path):
        os.remove(temp_corrupted_wav_path)
    if temp_created_wav and os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

    # 6. Assertions
    print("\n3. Running verification assertions...")
    
    # Assert 1: The corrupted phoneme's GoP score must fall below 40% threshold
    corrupted_score = corrupted_gop[corrupt_idx]["gop_prob"]
    print(f"Assertion 1 (Corrupted Phoneme '{target_phoneme}' GoP < 40%):")
    print(f"  Score: {corrupted_score:.2%}")
    assert corrupted_score < 0.40, f"Assertion failed: Corrupted phoneme '{target_phoneme}' maintained high GoP of {corrupted_score:.2%}"
    assert corrupted_gop[corrupt_idx]["is_correct"] == False, f"Assertion failed: Corrupted phoneme '{target_phoneme}' still flagged as correct."
    print("  ✅ Passed!")

    # Assert 2: Uncorrupted phonemes should maintain higher GoP scores (or at least remain above 40% if they were correct initially)
    print("Assertion 2 (Uncorrupted Phonemes maintain GoP accuracy):")
    uncorrupted_checked = 0
    for idx, detail in enumerate(corrupted_gop):
        if idx == corrupt_idx:
            continue
        # We only check uncorrupted phonemes that were highly correct in the baseline (>60%)
        baseline_score = gop_details[idx]["gop_prob"]
        if baseline_score >= 0.60:
            score = detail["gop_prob"]
            print(f"  Uncorrupted phoneme [{idx}] '{detail['phoneme']}': baseline={baseline_score:.2%}, corrupted_run={score:.2%}")
            assert score >= 0.40, f"Assertion failed: Uncorrupted phoneme '{detail['phoneme']}' dropped below 40% (GoP={score:.2%})"
            assert detail["is_correct"] == True, f"Assertion failed: Uncorrupted phoneme '{detail['phoneme']}' incorrectly flagged as false."
            uncorrupted_checked += 1
            
    print(f"  ✅ Passed! Verified {uncorrupted_checked} uncorrupted phonemes maintain high GoP scores.")

    print("\n" + "=" * 60)
    print("🎉 GOP SYSTEM VALIDATION PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

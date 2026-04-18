# run using mic
# python test_model.py --model_dir phoneme_model_out --duration 4.0
# existing wave file
# python test_model.py --model_dir phoneme_model_out --wav examples/test.wav
# with reference text (use quotes!)
# python test_model.py --duration 4.0 --ref_text "b ɪ k ə z"
# with word lookup
# python test_model.py --duration 4.0 --word because

import argparse
import sounddevice as sd
import soundfile as sf
import torch
import numpy as np
import os
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
from ScoreCalcs import PronunciationScorer
from audio_utils import AudioPreprocessor
from g2p.g2p_utils import G2PManager
import re

# G2P logic now handled by G2PManager in g2p/g2p_utils.py

def normalize_phonemes(phonemes):
    """Normalize phoneme representation for comparison"""
    # Remove stress markers and convert to uppercase for comparison
    normalized = []
    for p in phonemes:
        # Remove digits (stress markers)
        p_clean = re.sub(r'\d+', '', p)
        # Strip whitespace and convert to lowercase for consistency
        p_clean = p_clean.strip().lower()
        if p_clean and p_clean != '<pad>' and p_clean != '<unk>':
            normalized.append(p_clean)
    return normalized

def collapse_and_remove_pad(ids, pad_id):
    """Remove consecutive duplicates and padding tokens"""
    out = []
    prev = None
    for i in ids:
        if i == prev:
            continue
        prev = i
        if i == pad_id:
            continue
        out.append(int(i))
    return out

def format_phoneme_list(phonemes, max_width=70):
    """Format phoneme list with proper spacing"""
    return ' '.join(phonemes)

def print_section(title, char='='):
    """Print a formatted section header"""
    width = 70
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

if __name__ == "__main__":
    # G2PManager will be initialized inside main to handle custom dict paths if needed
    
    p = argparse.ArgumentParser(
        description="Test phoneme recognition model with pronunciation scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record from microphone
  python test_model.py --duration 4.0
  
  # Use existing audio file
  python test_model.py --wav examples/test.wav
  
  # With word lookup (automatic phoneme conversion)
  python test_model.py --duration 4.0 --word because
  
  # With manual phonemes (use quotes and spaces!)
  python test_model.py --duration 4.0 --ref_text "b ɪ k ə z"
  
  # Show available words
  python test_model.py --list_words | head -20
        """
    )
    p.add_argument("--model_dir", default="phoneme_model_out", 
                   help="Path to the trained model directory")
    p.add_argument("--wav", default=None, 
                   help="Optional wav file to infer; if not given, records from mic")
    p.add_argument("--duration", type=float, default=3.0, 
                   help="Microphone recording duration in seconds")
    p.add_argument("--ref_text", default=None, 
                   help='Reference phonemes (space-separated, e.g., "b ɪ k ə z")')
    p.add_argument("--word", default=None,
                   help='Word to look up in dictionary (e.g., "because")')
    p.add_argument("--list_words", action='store_true',
                   help='List all available words in dictionary')
    p.add_argument("--show_raw", action='store_true',
                   help='Show raw predictions before collapsing')
    
    args = p.parse_args()
    
    # Initialize G2P Manager
    g2p_manager = G2PManager()
    
    # Handle word list request
    if args.list_words:
        if g2p_manager.phoneme_dict:
            print(f"\n{len(g2p_manager.phoneme_dict)} words available:")
            for word, phonemes in sorted(g2p_manager.phoneme_dict.items())[:50]:
                print(f"  {word:15} → {' '.join(phonemes)}")
            if len(g2p_manager.phoneme_dict) > 50:
                print(f"  ... and {len(g2p_manager.phoneme_dict) - 50} more")
        else:
            print("No dictionary loaded.")
        exit(0)
    
    # Handle word lookup
    if args.word:
        word_lower = args.word.lower()
        if word_lower in g2p_manager.phoneme_dict:
            args.ref_text = ' '.join(g2p_manager.phoneme_dict[word_lower])
            print(f"✓ Word '{args.word}' → {args.ref_text}")
        else:
            print(f"✗ Word '{args.word}' not found in dictionary")
            # Suggest similar words
            similar = [w for w in g2p_manager.phoneme_dict.keys() if word_lower in w or w in word_lower][:5]
            if similar:
                print(f"  Similar words: {', '.join(similar)}")
            exit(1)
    
    # Validate ref_text format if provided
    if args.ref_text and not args.ref_text.strip():
        print("⚠ Warning: --ref_text is empty. Skipping pronunciation scoring.")
        args.ref_text = None
    
    # Load model and processor
    print(f"\nLoading model from {args.model_dir}...")
    try:
        from phoneme_embedder import Wav2Vec2PhonemeEmbedder
        processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
        # Attempt to load as custom embedder first, fallback to standard if it fails
        try:
            model = Wav2Vec2PhonemeEmbedder.from_pretrained(args.model_dir)
            print("✓ Loaded custom Wav2Vec2PhonemeEmbedder")
        except Exception:
            model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
            print("✓ Loaded standard Wav2Vec2ForCTC")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print(f"  Make sure '{args.model_dir}' exists and contains a valid model.")
        exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    model.to(device)
    model.eval()
    
    # Initialize Audio Preprocessor
    print("✓ Initializing Audio Preprocessor (FFT + Silero VAD)...")
    audio_prep = AudioPreprocessor(sr=16000)

    pad_id = processor.tokenizer.pad_token_id
    
    # Load phoneme mapping
    phoneme_map_path = os.path.join(args.model_dir, "phoneme2id.json")
    try:
        with open(phoneme_map_path, "r", encoding="utf8") as f:
            phoneme2id = json.load(f)
        id2phoneme = {int(v): k for k, v in phoneme2id.items()}
    except FileNotFoundError:
        print(f"⚠ Warning: {phoneme_map_path} not found. Using default mapping.")
        id2phoneme = {}
    
    # Load or record audio
    if args.wav:
        print(f"\nLoading audio from {args.wav}...")
        try:
            speech, sr = sf.read(args.wav)
            if len(speech.shape) > 1:
                speech = speech.mean(axis=1)
            if sr != 16000:
                print(f"  Resampling from {sr}Hz to 16000Hz...")
                speech = torchaudio.functional.resample(
                    torch.tensor(speech), sr, 16000
                ).numpy()
                sr = 16000
            print(f"✓ Loaded {len(speech)/sr:.2f}s of audio")
        except Exception as e:
            print(f"✗ Error loading audio file: {e}")
            exit(1)
    else:
        sr = 16000
        print(f"\n{'='*70}")
        print(f"🎤 Recording {args.duration}s from microphone...")
        print(f"{'='*70}")
        if args.word:
            print(f"Please say: '{args.word}'")
        elif args.ref_text:
            print(f"Target phonemes: {args.ref_text}")
        print("\n⏺  Recording NOW...")
        try:
            rec = sd.rec(
                int(args.duration * sr), 
                samplerate=sr, 
                channels=1, 
                dtype='float32'
            )
            sd.wait()
            speech = rec.squeeze()
            print("✓ Recording complete.\n")
        except Exception as e:
            print(f"✗ Error recording audio: {e}")
            exit(1)
    
    # Preprocess Audio (FFT Filter + VAD Trim)
    print("Cleaning audio (Applying FFT filter and VAD)...")
    original_len = len(speech)
    speech = audio_prep.preprocess(speech)
    print(f"✓ Audio cleaned. Length reduced from {original_len/sr:.2f}s to {len(speech)/sr:.2f}s")

    # Run inference
    print("Running inference...")
    inputs = processor(
        speech, 
        sampling_rate=sr, 
        return_tensors="pt", 
        padding=True
    )
    input_values = inputs.input_values.to(device)
    
    with torch.no_grad():
        outputs = model(input_values)
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs.logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    pred_phonemes_raw = [
        id2phoneme.get(int(i), '<unk>') for i in pred_ids[0]
    ]
    
    # Collapse repeated phonemes and remove padding
    collapsed = collapse_and_remove_pad(
        pred_ids[0].cpu().numpy().tolist(), 
        pad_id
    )
    phonemes_collapsed = [id2phoneme.get(i, '<unk>') for i in collapsed]
    
    # Show results
    print_section("PREDICTED PHONEMES")
    print(format_phoneme_list(phonemes_collapsed))
    
    if args.show_raw:
        print_section("RAW PREDICTIONS (before collapsing)", '-')
        # Show first 100 raw predictions
        raw_display = pred_phonemes_raw[:100]
        print(format_phoneme_list(raw_display))
        if len(pred_phonemes_raw) > 100:
            print(f"... and {len(pred_phonemes_raw) - 100} more frames")
    
    # Pronunciation scoring if reference text provided
    if args.ref_text:
        # Filter out padding tokens for valid predictions
        valid_pred_phonemes = [p for p in pred_phonemes_raw if p not in ['<pad>', '<unk>']]
        ref_phonemes_raw = args.ref_text.strip().split()
        
        # Normalize both for better comparison
        ref_phonemes_norm = normalize_phonemes(ref_phonemes_raw)
        pred_phonemes_norm = normalize_phonemes(valid_pred_phonemes)
        
        print_section("PRONUNCIATION ANALYSIS")
        
        if args.word:
            print(f"Target Word:  {args.word}")
        print(f"Reference:    {format_phoneme_list(ref_phonemes_raw)} ({len(ref_phonemes_raw)} phonemes)")
        print(f"Predicted:    {format_phoneme_list(valid_pred_phonemes[:len(ref_phonemes_raw)*3])} ({len(valid_pred_phonemes)} phonemes)")
        
        try:
            scorer = PronunciationScorer()
            duration = len(speech) / sr
            
            # Create timing estimates
            pred_times = [(i*0.02, (i+1)*0.02) for i in range(len(valid_pred_phonemes))]
            ref_times = [(i*duration/len(ref_phonemes_raw), (i+1)*duration/len(ref_phonemes_raw)) 
                        for i in range(len(ref_phonemes_raw))]
            
            results = scorer.compute_scores(
                pred_phonemes=valid_pred_phonemes,
                ref_phonemes=ref_phonemes_raw,
                pred_times=pred_times,
                ref_times=ref_times,
                pred_waveform=torch.tensor(speech).unsqueeze(0),
                ref_waveform=torch.tensor(speech).unsqueeze(0),
                sr=sr
            )
            
            # Display scores
            print("\n" + "─"*70)
            print("📊 SCORES")
            print("─"*70)
            
            accuracy = results['phoneme']
            if accuracy >= 0.8:
                status = "✓ Excellent"
            elif accuracy >= 0.6:
                status = "○ Good"
            elif accuracy >= 0.4:
                status = "△ Fair"
            else:
                status = "✗ Poor"
            
            print(f"  Phoneme Accuracy:  {accuracy:6.1%}  {status}")
            
            if 'duration' in results:
                dur_acc = results['duration']['accuracy']
                print(f"  Duration Match:    {dur_acc:6.1%}")
                print(f"  Duration Ratio:    {results['duration']['avg_ratio']:6.2f}x")
                print(f"  Timing Error:      {results['duration']['error_ms']:6.1f} ms")
            
            if 'pitch' in results:
                print(f"  Pitch Similarity:  {results['pitch']['similarity']:6.1%}")
                print(f"  Pitch Error:       {results['pitch']['error_hz']:6.1f} Hz")
            
            if 'stress' in results:
                print(f"  Stress Accuracy:   {results['stress']['accuracy']:6.1%}")
            
            # Error breakdown
            print("\n" + "─"*70)
            print("📋 ERROR BREAKDOWN")
            print("─"*70)
            stats = results['error_stats']
            total_errors = stats['sub'] + stats['ins'] + stats['del']
            print(f"  Substitutions:  {stats['sub']:3d}  (wrong phoneme)")
            print(f"  Insertions:     {stats['ins']:3d}  (extra phoneme)")
            print(f"  Deletions:      {stats['del']:3d}  (missing phoneme)")
            print(f"  Total Errors:   {total_errors:3d}")
            
            # Show alignment
            print("\n" + "─"*70)
            print("🔗 PHONEME ALIGNMENT")
            print("─"*70)
            print("    Predicted  →  Reference")
            print("  " + "─"*35)
            
            for i, (p, r) in enumerate(results['aligned_pairs'][:40], 1):
                if p == r:
                    symbol = "✓"
                elif p == '-':
                    symbol = "✗"  # deletion
                elif r == '-':
                    symbol = "+"  # insertion
                else:
                    symbol = "≠"  # substitution
                
                p_display = p if p != '-' else '___'
                r_display = r if r != '-' else '___'
                print(f"  {symbol} {p_display:>8}  →  {r_display:<8}")
            
            if len(results['aligned_pairs']) > 40:
                print(f"  ... and {len(results['aligned_pairs']) - 40} more")
        
        except Exception as e:
            print(f"\n✗ Error during pronunciation scoring: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70)
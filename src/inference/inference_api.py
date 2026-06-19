import os
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from src.utils.audio_utils import AudioPreprocessor
from src.eval.ScoreCalcs import PronunciationScorer
from src.g2p.g2p_utils import G2PManager
import json

# Global state
_model = None
_processor = None
_audio_prep = None
_scorer = None
_id2phoneme = {}
_g2p_manager = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_pipeline(model_dir: str):
    global _model, _processor, _audio_prep, _scorer, _id2phoneme, _g2p_manager
    if _model is not None:
        return
    
    # Resolve to absolute path so HuggingFace treats it as local, not a repo ID
    model_dir = os.path.abspath(model_dir)
    
    _g2p_manager = G2PManager()
    
    _processor = Wav2Vec2Processor.from_pretrained(model_dir)
    try:
        from src.models.phoneme_embedder import Wav2Vec2PhonemeEmbedder
        _model = Wav2Vec2PhonemeEmbedder.from_pretrained(model_dir)
    except Exception:
        _model = Wav2Vec2ForCTC.from_pretrained(model_dir)
        
    _model.to(device)
    _model.eval()
    
    _audio_prep = AudioPreprocessor(sr=16000)
    _scorer = PronunciationScorer()
    
    # Try loading phoneme2id.json, fallback to vocab.json if not found
    phoneme_map_path = os.path.join(model_dir, "phoneme2id.json")
    vocab_path = os.path.join(model_dir, "vocab.json")
    
    phoneme2id = None
    if os.path.exists(phoneme_map_path):
        try:
            with open(phoneme_map_path, "r", encoding="utf8") as f:
                phoneme2id = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {phoneme_map_path}: {e}")
            
    if phoneme2id is None and os.path.exists(vocab_path):
        try:
            with open(vocab_path, "r", encoding="utf8") as f:
                phoneme2id = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {vocab_path}: {e}")
            
    if phoneme2id:
        _id2phoneme = {int(v): k for k, v in phoneme2id.items()}
    else:
        _id2phoneme = {}

def run_inference(audio_path: str, target_word: str = None, target_phonemes: str = None, preprocess: bool = True) -> dict:
    global _model, _processor, _audio_prep, _scorer, _id2phoneme, _g2p_manager
    
    if _model is None:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline first.")
        
    print("[DEBUG run_inference] Processing target phonemes/word", flush=True)
    ref_phonemes_raw = []
    if target_word:
        # Use convert_sentence — handles multi-word input, dictionary lookup + neural fallback
        ref_phonemes_raw = _g2p_manager.convert_sentence(target_word)
        if not ref_phonemes_raw:
            raise ValueError(f"Could not generate phonemes for '{target_word}'.")
    elif target_phonemes:
        ref_phonemes_raw = target_phonemes.strip().split()
    else:
        raise ValueError("Either target_word or target_phonemes must be provided.")
        
    print("[DEBUG run_inference] Reading audio", flush=True)
    sr = 16000
    speech, out_sr = sf.read(audio_path)
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)
    if out_sr != 16000:
        import torchaudio
        speech = torchaudio.functional.resample(torch.tensor(speech), out_sr, 16000).numpy()
        
    if preprocess:
        print("[DEBUG run_inference] Preprocessing audio", flush=True)
        speech = _audio_prep.preprocess(speech)
    
    print("[DEBUG run_inference] Running processor", flush=True)
    inputs = _processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    
    print(f"[DEBUG run_inference] Running model on device: {device}. input_values shape: {input_values.shape}", flush=True)
    with torch.no_grad():
        outputs = _model(input_values)
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs.logits
            
    print(f"[DEBUG run_inference] Model prediction completed. Logits shape: {logits.shape}", flush=True)
    pred_ids = torch.argmax(logits, dim=-1)
    pred_phonemes_raw = [_id2phoneme.get(int(i), '<unk>') for i in pred_ids[0]]
    valid_pred_phonemes = [p for p in pred_phonemes_raw if p not in ['<pad>', '<unk>']]
    
    # Map target phonemes to IDs
    target_ids = _processor.tokenizer.convert_tokens_to_ids(ref_phonemes_raw)
    targets = torch.tensor([target_ids], dtype=torch.long, device=device)
    blank_id = _processor.tokenizer.pad_token_id or 0
    
    print(f"[DEBUG run_inference] Preparing for CTC forced align. target_ids length: {len(target_ids)}", flush=True)
    # Run CTC forced alignment
    intervals = _scorer.ctc_forced_align(logits, targets, blank_id=blank_id)
    
    print("[DEBUG run_inference] CTC forced align completed. Computing GoP", flush=True)
    # Run GoP Scorer
    gop_details = _scorer.compute_gop(logits, targets, intervals, ref_phonemes_raw)
    
    print("[DEBUG run_inference] GoP computed. Computing final scores", flush=True)
    duration = len(speech) / sr
    pred_times = [(i*0.02, (i+1)*0.02) for i in range(len(valid_pred_phonemes))]
    ref_times = [(i*duration/len(ref_phonemes_raw), (i+1)*duration/len(ref_phonemes_raw)) 
                for i in range(len(ref_phonemes_raw))]
                
    results = _scorer.compute_scores(
        pred_phonemes=valid_pred_phonemes,
        ref_phonemes=ref_phonemes_raw,
        pred_times=pred_times,
        ref_times=ref_times,
        pred_waveform=torch.tensor(speech).unsqueeze(0),
        ref_waveform=torch.tensor(speech).unsqueeze(0),
        sr=sr
    )
    
    results["gop_details"] = gop_details
    return results

import os
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from research.audio_utils import AudioPreprocessor
from research.ScoreCalcs import PronunciationScorer
from research.g2p.g2p_utils import G2PManager
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
        from research.phoneme_embedder import Wav2Vec2PhonemeEmbedder
        _model = Wav2Vec2PhonemeEmbedder.from_pretrained(model_dir)
    except Exception:
        _model = Wav2Vec2ForCTC.from_pretrained(model_dir)
        
    _model.to(device)
    _model.eval()
    
    _audio_prep = AudioPreprocessor(sr=16000)
    _scorer = PronunciationScorer()
    
    phoneme_map_path = os.path.join(model_dir, "phoneme2id.json")
    try:
        with open(phoneme_map_path, "r", encoding="utf8") as f:
            phoneme2id = json.load(f)
        _id2phoneme = {int(v): k for k, v in phoneme2id.items()}
    except FileNotFoundError:
        _id2phoneme = {}

def run_inference(audio_path: str, target_word: str = None, target_phonemes: str = None) -> dict:
    global _model, _processor, _audio_prep, _scorer, _id2phoneme, _g2p_manager
    
    if _model is None:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline first.")
        
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
        
    sr = 16000
    speech, out_sr = sf.read(audio_path)
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)
    if out_sr != 16000:
        import torchaudio
        speech = torchaudio.functional.resample(torch.tensor(speech), out_sr, 16000).numpy()
        
    speech = _audio_prep.preprocess(speech)
    
    inputs = _processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    
    with torch.no_grad():
        outputs = _model(input_values)
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs.logits
            
    pred_ids = torch.argmax(logits, dim=-1)
    pred_phonemes_raw = [_id2phoneme.get(int(i), '<unk>') for i in pred_ids[0]]
    valid_pred_phonemes = [p for p in pred_phonemes_raw if p not in ['<pad>', '<unk>']]
    
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
    
    return results

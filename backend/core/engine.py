from research.inference_api import init_pipeline, run_inference

class ASREngine:
    def __init__(self, model_dir: str):
        init_pipeline(model_dir)
        
    def evaluate(self, audio_path: str, target_word: str = None, target_phonemes: str = None) -> dict:
        return run_inference(
            audio_path=audio_path,
            target_word=target_word,
            target_phonemes=target_phonemes
        )

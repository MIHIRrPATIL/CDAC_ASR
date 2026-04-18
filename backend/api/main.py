import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from backend.core.engine import ASREngine
from api.auth_routes import router as auth_router
from database import get_db
from models import AudioEntry, User
from auth import get_current_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pronunciation Scoring API")
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
def startup_event():
    global engine
    # Safe fallback if environment variable isn't set
    model_dir = os.getenv("ASR_MODEL_DIR", "models/trained_models/1_epoch")
    logger.info(f"Loading ASR Engine with model from {model_dir}")
    try:
        engine = ASREngine(model_dir=model_dir)
        logger.info("ASR Engine successfully initialized.")
    except Exception as e:
        logger.error(f"Failed to load ASR engine: {e}")
        
@app.post("/analyze")
async def analyze_audio(
    audio_file: UploadFile = File(...),
    target_word: str = Form(None),
    target_phonemes: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="ASR Engine not loaded properly due to startup error.")
    
    if not target_word and not target_phonemes:
        raise HTTPException(status_code=400, detail="Must provide either 'target_word' or 'target_phonemes'.")
        
    # Determine the file suffix (e.g. .wav)
    suffix = ".wav" if audio_file.filename and audio_file.filename.endswith(".wav") else ""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    try:
        content = await audio_file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Bridge to the research pipeline interface
        results = engine.evaluate(
            audio_path=temp_file.name,
            target_word=target_word,
            target_phonemes=target_phonemes
        )
        
        # Package and decouple metrics vs alignments logic
        scores = {k: v for k, v in results.items() if k not in ['error_stats', 'aligned_pairs']}
        analysis = {
            'error_stats': results.get('error_stats', {}),
            'aligned_pairs': results.get('aligned_pairs', [])
        }
        # --- DB Persistence ---
        try:
            # Extract scores for DB
            phoneme_score = scores.get('phoneme_score', 0.0)
            duration_score = scores.get('duration_score', 0.0)
            
            pitch_data = scores.get('pitch_score')
            if isinstance(pitch_data, dict):
                pitch_score = pitch_data.get('similarity', 0.0)
            elif isinstance(pitch_data, (float, int)):
                pitch_score = float(pitch_data)
            else:
                pitch_score = 0.0
                
            stress_score = scores.get('stress_score', 0.0)
            
            # Simple average for overall score
            overall_score = (phoneme_score + duration_score + pitch_score + stress_score) / 4.0

            new_entry = AudioEntry(
                user_id=current_user.id,
                target_word=target_word or "custom_phonemes",
                overall_score=overall_score,
                phoneme_score=phoneme_score,
                duration_score=duration_score,
                pitch_score=pitch_score,
                stress_score=stress_score,
                pitch_trajectory=scores.get('pitch_trajectory'),
                phoneme_alignment=analysis.get('aligned_pairs')
            )
            db.add(new_entry)
            db.commit()
            logger.info(f"Saved audio entry for user {current_user.id}")
        except Exception as db_err:
            logger.error(f"Failed to save to DB: {db_err}")
            # We don't raise here, still return results to user
        
        return {
            "scores": scores,
            "analysis": analysis,
            "feedback": None
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed internally.")
    finally:
        # Secure cleanup
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

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
    model_dir = os.getenv("ASR_MODEL_DIR", "MihirRPatil/nptel-asr-phoneme-v3")
    logger.info(f"Loading ASR Engine with model from {model_dir}")
    try:
        engine = ASREngine(model_dir=model_dir)
        logger.info("ASR Engine successfully initialized.")
    except Exception as e:
        logger.error(f"Failed to load ASR engine: {e}")

def generate_actionable_feedback(results: dict) -> list:
    feedback = []
    
    # 1. Phoneme errors analysis
    error_stats = results.get("error_stats", {})
    subs = error_stats.get("sub", 0)
    ins = error_stats.get("ins", 0)
    dels = error_stats.get("del", 0)
    
    # Overall statement
    total_errors = subs + ins + dels
    if total_errors == 0:
        feedback.append("Excellent pronunciation! All phonemes aligned perfectly.")
    else:
        feedback.append(f"Pronunciation analysis detected {total_errors} variant sound{'s' if total_errors > 1 else ''}: {subs} substitution{'s' if subs != 1 else ''}, {dels} omission{'s' if dels != 1 else ''}, and {ins} extra sound{'s' if ins != 1 else ''}.")

    # 2. Specific segment level recommendations from alignment pairs
    aligned_pairs = results.get("aligned_pairs", [])
    
    # Find up to 3 specific error instances to guide the user
    error_count = 0
    for idx, (spoken, expected) in enumerate(aligned_pairs):
        if error_count >= 3:
            break
            
        if spoken == "-": # Deletion
            feedback.append(f"• You omitted the expected sound '{expected}' at position {idx+1}. Try to fully articulate it.")
            error_count += 1
        elif expected == "-": # Insertion
            feedback.append(f"• You added an extra sound '{spoken}' at position {idx+1}. Practice transitioning without this extra sound.")
            error_count += 1
        elif spoken != expected: # Substitution
            feedback.append(f"• You pronounced '{expected}' as '{spoken}' at position {idx+1}. Try adjusting your tongue shape to match the target sound.")
            error_count += 1

    # 3. Prosody and Pitch recommendations
    pitch_data = results.get("pitch", {})
    pitch_similarity = pitch_data.get("similarity", 1.0) if isinstance(pitch_data, dict) else float(pitch_data)
    if pitch_similarity < 0.60:
        feedback.append("• Pitch Curve: Your intonation curve differs significantly from the reference. Focus on native-like sentence stress and rising/falling tones.")
    elif pitch_similarity < 0.80:
        feedback.append("• Intonation: Good effort! Try matching the pitch rises and falls on the stressed vowel syllables to sound more natural.")
        
    duration_data = results.get("duration", {})
    duration_accuracy = duration_data.get("accuracy", 1.0) if isinstance(duration_data, dict) else float(duration_data)
    if duration_accuracy < 0.65:
        feedback.append("• Rhythm: The timing of some phonemes is too rushed or held too long. Practice syllable-by-syllable timing.")
        
    return feedback

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
        feedback_list = generate_actionable_feedback(results)
        # --- DB Persistence ---
        try:
            # Extract scores for DB using the correct keys from research/ScoreCalcs.py
            phoneme_score = scores.get('phoneme', 0.0)
            
            duration_data = scores.get('duration')
            if isinstance(duration_data, dict):
                duration_score = duration_data.get('accuracy', 0.0)
            elif isinstance(duration_data, (float, int)):
                duration_score = float(duration_data)
            else:
                duration_score = 0.0
            
            pitch_data = scores.get('pitch')
            if isinstance(pitch_data, dict):
                pitch_score = pitch_data.get('similarity', 0.0)
            elif isinstance(pitch_data, (float, int)):
                pitch_score = float(pitch_data)
            else:
                pitch_score = 0.0
                
            stress_data = scores.get('stress')
            if isinstance(stress_data, dict):
                stress_score = stress_data.get('accuracy', 0.0)
            elif isinstance(stress_data, (float, int)):
                stress_score = float(stress_data)
            else:
                stress_score = 0.0
            
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
            "feedback": feedback_list
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

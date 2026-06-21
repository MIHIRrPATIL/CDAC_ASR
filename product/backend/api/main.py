import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.core.engine import ASREngine
from api.auth_routes import router as auth_router
from api.features_routes import router as features_router
from database import get_db
from prisma import Json
from prisma.models import User
from auth import get_current_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pronunciation Scoring API")
app.include_router(auth_router)
app.include_router(features_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    # Initialize connection to Prisma
    from database import db as prisma_db
    if not prisma_db.is_connected():
        await prisma_db.connect()
        
    # Safe fallback if environment variable isn't set
    model_dir = os.getenv("ASR_MODEL_DIR", "MihirRPatil/nptel-asr-phoneme-v3")
    logger.info(f"Loading ASR Engine with model from {model_dir}")
    try:
        engine = ASREngine(model_dir=model_dir)
        logger.info("ASR Engine successfully initialized.")
    except Exception as e:
        logger.error(f"Failed to load ASR engine: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    from database import db as prisma_db
    if prisma_db.is_connected():
        await prisma_db.disconnect()

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

def transcode_to_wav(input_path: str) -> str:
    """
    Transcodes any audio file to 16kHz mono WAV using ffmpeg.
    Returns the path to the temporary output WAV file.
    """
    import subprocess
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_out_path = temp_out.name
    temp_out.close()
    
    try:
        # Run ffmpeg to convert input to 16kHz mono PCM 16-bit WAV
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            temp_out_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg transcoding failed. Stderr: {result.stderr}")
            return input_path
            
        logger.info(f"Successfully transcoded {input_path} to {temp_out_path}")
        return temp_out_path
    except Exception as e:
        logger.error(f"Error during ffmpeg transcoding: {e}")
        return input_path

def analyze_words_pronunciation(target_word: str, aligned_pairs: list) -> list:
    """
    Groups the flat aligned_pairs into individual words based on the expected phonemes.
    """
    if not target_word:
        return []
        
    try:
        from src.g2p.g2p_utils import G2PManager
        g2p = G2PManager()
        words = g2p.tokenize(target_word)
        
        words_expected_phonemes = []
        for word in words:
            words_expected_phonemes.append((word, g2p.convert_word(word)))
            
        words_analysis = []
        for word, expected_phns in words_expected_phonemes:
            words_analysis.append({
                "word": word,
                "phonemes": expected_phns,
                "aligned_pairs": [],
                "accuracy": 1.0,
                "error_stats": {"sub": 0, "ins": 0, "del": 0},
                "status": "correct"
            })
            
        if not words_analysis:
            return []
            
        curr_word_idx = 0
        curr_phn_idx = 0
        
        for spoken, expected in aligned_pairs:
            if curr_word_idx >= len(words_analysis):
                target_word_idx = len(words_analysis) - 1
                words_analysis[target_word_idx]["aligned_pairs"].append([spoken, expected])
                continue
                
            if expected == "-":
                words_analysis[curr_word_idx]["aligned_pairs"].append([spoken, expected])
            else:
                words_analysis[curr_word_idx]["aligned_pairs"].append([spoken, expected])
                curr_phn_idx += 1
                while (curr_word_idx < len(words_analysis) and 
                       curr_phn_idx >= len(words_analysis[curr_word_idx]["phonemes"])):
                    curr_word_idx += 1
                    curr_phn_idx = 0
                    
        for word_data in words_analysis:
            aligned = word_data["aligned_pairs"]
            correct = sum(1 for p, r in aligned if p == r)
            total_ref = sum(1 for _, r in aligned if r != '-')
            
            accuracy = (correct / total_ref) if total_ref > 0 else 0.0
            word_data["accuracy"] = accuracy
            
            sub = sum(1 for p, r in aligned if p != '-' and r != '-' and p != r)
            ins = sum(1 for p, r in aligned if r == '-')
            deletion = sum(1 for p, r in aligned if p == '-')
            word_data["error_stats"] = {"sub": sub, "ins": ins, "del": deletion}
            
            if sub > 0 or ins > 0 or deletion > 0:
                word_data["status"] = "incorrect"
            else:
                word_data["status"] = "correct"
                
        return words_analysis
    except Exception as e:
        logger.error(f"Error in analyze_words_pronunciation: {e}", exc_info=True)
        return []

@app.post("/analyze")
async def analyze_audio(
    audio_file: UploadFile = File(...),
    target_word: str = Form(None),
    target_phonemes: str = Form(None),
    db = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="ASR Engine not loaded properly due to startup error.")
    
    if not target_word and not target_phonemes:
        raise HTTPException(status_code=400, detail="Must provide either 'target_word' or 'target_phonemes'.")
        
    suffix = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ""
    temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_raw_path = temp_raw.name
    
    temp_wav_path = None
    try:
        content = await audio_file.read()
        temp_raw.write(content)
        temp_raw.close()
        
        temp_wav_path = transcode_to_wav(temp_raw_path)
        
        results = engine.evaluate(
            audio_path=temp_wav_path,
            target_word=target_word,
            target_phonemes=target_phonemes
        )
        
        scores = {k: v for k, v in results.items() if k not in ['error_stats', 'aligned_pairs']}
        
        words_analysis = []
        if target_word:
            words_analysis = analyze_words_pronunciation(target_word, results.get('aligned_pairs', []))
            
        analysis = {
            'error_stats': results.get('error_stats', {}),
            'aligned_pairs': results.get('aligned_pairs', []),
            'words_analysis': words_analysis
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

            pitch_trajectory = None
            pitch_data = scores.get('pitch')
            if isinstance(pitch_data, dict):
                pitch_trajectory = pitch_data.get('trajectory')
            phoneme_alignment = analysis.get('aligned_pairs')

            db_data = {
                "user": {"connect": {"id": current_user.id}},
                "targetWord": target_word or "custom_phonemes",
                "overallScore": float(overall_score),
                "phonemeScore": float(phoneme_score),
                "durationScore": float(duration_score),
                "pitchScore": float(pitch_score),
                "stressScore": float(stress_score),
            }

            if pitch_trajectory is not None:
                db_data["pitchTrajectory"] = Json(pitch_trajectory)
            if phoneme_alignment is not None:
                db_data["phonemeAlignment"] = Json(phoneme_alignment)

            await db.audioentry.create(data=db_data)
            logger.info(f"Saved audio entry for user {current_user.id}")
        except Exception as db_err:
            logger.error(f"Failed to save to DB: {db_err}")
            # We don't raise here, still return results to user
        
        return {
            "scores": scores,
            "analysis": analysis,
            "feedback": feedback_list,
            "target_word": target_word or "custom_phonemes"
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed internally.")
    finally:
        # Secure cleanup
        try:
            if os.path.exists(temp_raw_path):
                os.remove(temp_raw_path)
        except Exception as e:
            logger.error(f"Error removing temp_raw: {e}")
            
        try:
            if temp_wav_path and temp_wav_path != temp_raw_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception as e:
            logger.error(f"Error removing temp_wav: {e}")

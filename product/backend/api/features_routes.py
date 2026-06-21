import os
import logging
import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from database import get_db
from prisma.models import User
from auth import get_current_user
from services.tts import generate_reference_audio
from services.llm import (
    generate_weakness_targeted_paragraph,
    generate_roleplay_response,
    start_roleplay_conversation,
    generate_custom_drills,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["features"])

# ──── Pydantic schemas ────
class ReviewRequest(BaseModel):
    sr_id: str
    overall_score: float

class AddSRRequest(BaseModel):
    word: str
    phonemes: str

class ListCreateRequest(BaseModel):
    title: str
    description: str = None

class EntryCreateRequest(BaseModel):
    word: str
    phonemes: str = None

class RoleplayRequest(BaseModel):
    dialogue_history: list[dict]
    scenario: str = None

class StartRoleplayRequest(BaseModel):
    scenario: str = None

class TextGenerateRequest(BaseModel):
    topic: str = None

class DrillsGenerateRequest(BaseModel):
    prompt: str = None

# ──── 1. TTS Reference Audio Generator ────
@router.get("/tts/generate")
def get_tts(text: str = Query(...), slow: bool = Query(False)):
    """Generates and serves a cached reference audio clip using gTTS."""
    try:
        filepath = generate_reference_audio(text, slow)
        return FileResponse(filepath, media_type="audio/mp3", filename=os.path.basename(filepath))
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate text-to-speech audio.")

# ──── 2. Dashboard Statistics & Heatmap ────
@router.get("/dashboard/stats")
async def get_dashboard_stats(db = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Calculates active accuracy history, streaks, and phoneme heatmaps from DB entries."""
    try:
        # Fetch entries
        entries = await db.audioentry.find_many(
            where={"userId": current_user.id},
            order={"createdAt": "desc"},
            take=30
        )
        
        # Calculate recent scores trend
        scores_history = []
        for e in reversed(entries):
            scores_history.append({
                "date": e.createdAt.strftime("%b %d"),
                "score": int(e.overallScore * 100)
            })
            
        # Calculate phoneme alignment error matrix/heatmap
        phoneme_stats = {}
        for e in entries:
            alignment = e.phonemeAlignment # list of [spoken, expected]
            if not alignment:
                continue
            for pair in alignment:
                if len(pair) == 2:
                    spoken, expected = pair[0], pair[1]
                    if expected == "-" or expected == "":
                        continue # Skip insertion baseline
                        
                    if expected not in phoneme_stats:
                        phoneme_stats[expected] = {"correct": 0, "total": 0}
                        
                    phoneme_stats[expected]["total"] += 1
                    if spoken == expected:
                        phoneme_stats[expected]["correct"] += 1
                        
        heatmap = []
        for phoneme, stats in phoneme_stats.items():
            accuracy = stats["correct"] / stats["total"]
            heatmap.append({
                "phoneme": phoneme,
                "accuracy": int(accuracy * 100),
                "total_practiced": stats["total"]
            })
            
        # Sort heatmap by lowest accuracy first (weaknesses)
        heatmap.sort(key=lambda x: x["accuracy"])
        
        # Aggregate baseline stats
        overall_accuracy = int(sum(e.overallScore for e in entries) / len(entries) * 100) if entries else 0
        
        return {
            "overall_accuracy": overall_accuracy,
            "practice_seconds": current_user.totalPracticeSeconds,
            "daily_streak": current_user.dailyStreak,
            "global_rank": int(current_user.globalRankScore),
            "history": scores_history,
            "heatmap": heatmap[:12] # Top 12 weakest phonemes
        }
    except Exception as e:
        logger.error(f"Failed to compile dashboard stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard data.")

# ──── 3. Spaced Repetition Queue & Reviews ────
@router.get("/spaced-repetition/queue")
async def get_sr_queue(db = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Retrieves all flashcards in queue due for practice."""
    now = datetime.datetime.now(datetime.timezone.utc)
    queue = await db.spacedrepetition.find_many(
        where={
            "userId": current_user.id,
            "nextReviewAt": {"lte": now}
        },
        order={"nextReviewAt": "asc"}
    )
    return queue

@router.post("/spaced-repetition/add")
async def add_sr_card(req: AddSRRequest, db = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Manually adds a new word card to the spaced repetition deck."""
    now = datetime.datetime.now(datetime.timezone.utc)
    card = await db.spacedrepetition.create(
        data={
            "userId": current_user.id,
            "word": req.word,
            "phonemes": req.phonemes,
            "nextReviewAt": now
        }
    )
    return card

@router.post("/spaced-repetition/review")
async def review_sr_card(req: ReviewRequest, db = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Applies SM-2 scheduling algorithm to update flashcard review interval."""
    card = await db.spacedrepetition.find_unique(where={"id": req.sr_id})
    if not card or card.userId != current_user.id:
        raise HTTPException(status_code=404, detail="Flashcard not found.")
        
    # Map percentage score to SM-2 quality grade (0 to 5)
    score = req.overall_score * 100
    if score >= 90:
        q = 5
    elif score >= 80:
        q = 4
    elif score >= 70:
        q = 3
    elif score >= 50:
        q = 2
    elif score >= 30:
        q = 1
    else:
        q = 0
        
    repetitions = card.repetitions
    interval = card.interval
    ease_factor = card.easeFactor
    
    # SM-2 Scheduling Logic
    if q >= 3:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = int(interval * ease_factor)
        repetitions += 1
    else:
        repetitions = 0
        interval = 1
        
    ease_factor = ease_factor + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    if ease_factor < 1.3:
        ease_factor = 1.3
        
    next_review = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=interval)
    
    updated = await db.spacedrepetition.update(
        where={"id": req.sr_id},
        data={
            "repetitions": repetitions,
            "interval": interval,
            "easeFactor": ease_factor,
            "nextReviewAt": next_review
        }
    )
    return updated

# ──── 4. Custom Word Lists CRUD ────
@router.get("/lists")
async def get_word_lists(db = Depends(get_db), current_user: User = Depends(get_current_user)):
    lists = await db.customwordlist.find_many(
        where={"userId": current_user.id},
        include={"entries": True}
    )
    return lists

@router.post("/lists")
async def create_word_list(req: ListCreateRequest, db = Depends(get_db), current_user: User = Depends(get_current_user)):
    new_list = await db.customwordlist.create(
        data={
            "userId": current_user.id,
            "title": req.title,
            "description": req.description
        }
    )
    return new_list

@router.delete("/lists/{list_id}")
async def delete_word_list(list_id: str, db = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Check ownership
    lst = await db.customwordlist.find_unique(where={"id": list_id})
    if not lst or lst.userId != current_user.id:
        raise HTTPException(status_code=404, detail="Word list not found.")
        
    await db.customwordlist.delete(where={"id": list_id})
    return {"status": "success"}

@router.post("/lists/{list_id}/entries")
async def add_list_entry(list_id: str, req: EntryCreateRequest, db = Depends(get_db), current_user: User = Depends(get_current_user)):
    lst = await db.customwordlist.find_unique(where={"id": list_id})
    if not lst or lst.userId != current_user.id:
        raise HTTPException(status_code=404, detail="Word list not found.")
        
    entry = await db.wordlistentry.create(
        data={
            "listId": list_id,
            "word": req.word,
            "phonemes": req.phonemes
        }
    )
    return entry

# ──── 5. AI Tutor OpenRouter Endpoints ────
@router.post("/ai/generate-text")
async def get_ai_paragraph(req: TextGenerateRequest, db = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Finds user's lowest scoring phonemes from alignment history and generates custom paragraph."""
    # 1. Fetch recent user weaknesses
    entries = await db.audioentry.find_many(
        where={"userId": current_user.id},
        order={"createdAt": "desc"},
        take=15
    )
    
    phoneme_stats = {}
    for e in entries:
        alignment = e.phonemeAlignment
        if not alignment:
            continue
        for pair in alignment:
            if len(pair) == 2:
                spoken, expected = pair[0], pair[1]
                if expected == "-" or expected == "":
                    continue
                if expected not in phoneme_stats:
                    phoneme_stats[expected] = {"correct": 0, "total": 0}
                phoneme_stats[expected]["total"] += 1
                if spoken == expected:
                    phoneme_stats[expected]["correct"] += 1
                    
    weak_phonemes = []
    for p, stats in phoneme_stats.items():
        accuracy = stats["correct"] / stats["total"]
        if accuracy < 0.70: # Accuracy threshold
            weak_phonemes.append(p)
            
    # Sort weak phonemes
    weak_phonemes = weak_phonemes[:5]
    
    # 2. Call generator
    paragraph = await generate_weakness_targeted_paragraph(weak_phonemes, req.topic)
    return {
        "paragraph": paragraph,
        "targeted_phonemes": weak_phonemes
    }

@router.post("/ai/roleplay")
async def post_roleplay(req: RoleplayRequest, current_user: User = Depends(get_current_user)):
    """Handles OpenRouter conversation turn-taking."""
    result = await generate_roleplay_response(req.dialogue_history, req.scenario)
    return result

@router.post("/ai/start-roleplay")
async def post_start_roleplay(req: StartRoleplayRequest, current_user: User = Depends(get_current_user)):
    """Starts a new roleplay conversation based on a chosen scenario."""
    result = await start_roleplay_conversation(req.scenario)
    return result

@router.post("/ai/generate-drills")
async def post_generate_drills(req: DrillsGenerateRequest, current_user: User = Depends(get_current_user)):
    """Generates custom minimal pair drills targetting user phonemes or a sound prompt."""
    result = await generate_custom_drills(req.prompt)
    return result

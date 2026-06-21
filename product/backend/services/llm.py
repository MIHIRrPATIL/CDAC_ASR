import os
import httpx
import logging

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it:free"

def get_headers():
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "CDAC ASR Pronunciation Academy",
    }

async def generate_weakness_targeted_paragraph(weak_phonemes: list[str], topic: str = None) -> str:
    """Generates a short paragraph (2-3 sentences) heavily incorporating the user's weak phonemes."""
    if not weak_phonemes:
        phonemes_str = "r, th, s, z"
    else:
        phonemes_str = ", ".join(weak_phonemes)
        
    topic_str = f"about '{topic}'" if topic else "about general daily life"
    
    prompt = (
        f"You are a speech therapist and language coach. Generate a short practice reading paragraph (exactly 2 to 3 sentences) {topic_str}.\n"
        f"The paragraph MUST heavily repeat words containing these target IPA/ARPAbet sounds: {phonemes_str}.\n"
        f"Keep the language natural, easy to read, and clear.\n"
        f"Return ONLY the raw reading text, with no headers, explanations, or quotes."
    )
    
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful speech therapist assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(OPENROUTER_URL, json=payload, headers=get_headers())
            if response.status_code == 200:
                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()
                # Clean up any surrounding quotes
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                return text
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Failed to query OpenRouter: {e}")
        
    # Standard fallback paragraph if API fails
    return "The red roses grew rapidly around the rural riverbank. Through the thick weeds, we thought we saw three thin threads."

async def generate_roleplay_response(dialogue_history: list[dict], scenario: str = None) -> dict:
    """
    Given a chat history and scenario, generates the next conversational turn.
    Returns: {'response': str, 'corrections': str, 'suggested_replies': [str, str, str]}
    """
    import json
    import random

    if not scenario:
        scenario = "A friendly job interview for a software developer position"

    # Count how many turns have occurred for awareness
    turn_count = len(dialogue_history)
    user_turns = sum(1 for m in dialogue_history if m.get("is_user"))

    # Build a short recap of recent topics so the LLM can track conversation flow
    recent_user_lines = [m.get("text", "") for m in dialogue_history if m.get("is_user")][-3:]
    recent_ava_lines = [m.get("text", "") for m in dialogue_history if not m.get("is_user")][-3:]

    system_prompt = (
        f"You are Ava — a warm, expressive conversational partner in an English pronunciation practice app.\n"
        f"\n"
        f"SCENARIO: {scenario}\n"
        f"TURN COUNT: This is turn #{turn_count + 1}. The user has spoken {user_turns} times.\n"
        f"\n"
        f"YOUR PERSONALITY:\n"
        f"- Friendly, curious, and encouraging. You react genuinely to what the user says.\n"
        f"- You stay in character for the scenario (e.g. if it's a café scene, you're the barista/friend).\n"
        f"- You share relatable details, ask follow-up questions, and move the conversation forward naturally.\n"
        f"\n"
        f"CRITICAL RULES:\n"
        f"- NEVER repeat a line you already said. Your recent lines were: {json.dumps(recent_ava_lines)}\n"
        f"- NEVER say generic filler like 'That's very interesting' or 'Could you tell me more'. Be SPECIFIC.\n"
        f"- React to the ACTUAL CONTENT of the user's message. Reference details they mentioned.\n"
        f"- Keep your reply to 1-3 sentences. Be conversational, not robotic.\n"
        f"- Advance the conversation: introduce new sub-topics, share a personal anecdote, or ask a specific question.\n"
        f"\n"
        f"CONVERSATION FLOW GUIDE (adapt to scenario):\n"
        f"- Turns 1-3: Introductions, establish context, warm up\n"
        f"- Turns 4-6: Go deeper into topics, share stories, react with emotion\n"
        f"- Turns 7-9: Explore new angles, bring humor or surprise, build rapport\n"
        f"- Turns 10+: Wind down naturally or pivot to a new fun topic\n"
        f"\n"
        f"SUGGESTED REPLIES: Generate 3 diverse options the user could say next:\n"
        f"  1. A natural continuation (matching the scenario tone)\n"
        f"  2. A casual/fun alternative response\n"
        f"  3. A question or curiosity-driven response\n"
        f"Each suggestion should be 1-2 sentences, natural sounding, and DIFFERENT from each other.\n"
        f"\n"
        f"OUTPUT FORMAT — strict JSON, no extra text:\n"
        f"{{\n"
        f"  \"response\": \"your next conversational line as Ava\",\n"
        f"  \"corrections\": \"brief friendly grammar/vocabulary feedback on the user's last message, or empty string if perfect\",\n"
        f"  \"suggested_replies\": [\"option 1\", \"option 2\", \"option 3\"]\n"
        f"}}"
    )

    # Send more context to the LLM (up to 12 messages) for better flow
    messages = [{"role": "system", "content": system_prompt}]
    for msg in dialogue_history[-12:]:
        role = "user" if msg.get("is_user") else "assistant"
        messages.append({"role": role, "content": msg.get("text", "")})

    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.85,
    }

    BACKUP_MODELS = [
        "openai/gpt-oss-120b:free",
        "liquid/lfm-2.5-1.2b-instruct:free",
        "cohere/north-mini-code:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ]

    # Try primary model, then fallback to backup models
    models_to_try = [DEFAULT_MODEL] + BACKUP_MODELS
    last_error = None

    for model in models_to_try:
        payload["model"] = model
        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(OPENROUTER_URL, json=payload, headers=get_headers())
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    parsed = json.loads(content)
                    # Validate the response has required keys
                    if "response" in parsed and "suggested_replies" in parsed:
                        if not parsed.get("corrections"):
                            parsed["corrections"] = ""
                        return parsed
                    else:
                        logger.warning(f"Model {model} returned malformed JSON: {content[:200]}")
                        continue
                elif response.status_code == 429:
                    logger.warning(f"Model {model} rate limited (429), trying next...")
                    continue
                else:
                    logger.error(f"OpenRouter API error ({model}): {response.status_code} - {response.text[:200]}")
                    continue
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error from {model}: {e}")
            continue
        except Exception as e:
            last_error = e
            logger.error(f"Failed to query OpenRouter roleplay ({model}): {e}")
            continue

    # Varied contextual fallbacks instead of always the same line
    logger.error(f"All models failed for roleplay. Last error: {last_error}")
    last_user_msg = ""
    for m in reversed(dialogue_history):
        if m.get("is_user"):
            last_user_msg = m.get("text", "")
            break

    fallback_responses = [
        f"Oh nice! I love hearing about that. So what happened next?",
        f"Ha, that's a great point! Reminds me of something similar. What else is on your mind?",
        f"I totally get what you mean. So, switching gears a little — what's something exciting you've been up to lately?",
        f"That sounds like quite an experience! I'm curious, how did you feel about it afterward?",
        f"You know, I was just thinking about something like that the other day. What made you bring it up?",
    ]
    fallback_suggestions = [
        [
            "Well, it all started when I decided to try something new.",
            "Honestly, it was one of those moments that just stuck with me.",
            "What about you? Have you ever experienced something like that?",
        ],
        [
            "I think the best part was how unexpected everything turned out.",
            "Ha, that reminds me of a funny story actually!",
            "I'm curious — what would you have done in that situation?",
        ],
        [
            "Let me think about that for a moment... Actually, yes!",
            "That's a good question. I'd say it depends on the situation.",
            "To be honest, I've been meaning to talk about something related to that.",
        ],
    ]

    idx = random.randint(0, len(fallback_responses) - 1)
    return {
        "response": fallback_responses[idx],
        "corrections": "",
        "suggested_replies": fallback_suggestions[idx % len(fallback_suggestions)],
    }

async def start_roleplay_conversation(scenario: str) -> dict:
    """
    Starts a new roleplay conversation based on the user's chosen scenario.
    Returns: {"response": str, "suggested_replies": [str, str, str]}
    """
    import json

    if not scenario:
        scenario = "A friendly job interview for a software developer position"
        
    system_prompt = (
        f"You are Ava — a warm, expressive conversational partner in an English pronunciation practice app.\n"
        f"\n"
        f"SCENARIO: {scenario}\n"
        f"\n"
        f"You must start the conversation with an engaging, in-character opening line (1-2 sentences).\n"
        f"Be specific to the scenario — don't just say 'Hello, let's practice'. Instead, immerse yourself.\n"
        f"For example, if the scenario is a café, say something like 'Hey! Welcome in, it's so good to see you! I already grabbed us a table by the window.'\n"
        f"\n"
        f"Then provide 3 diverse suggested responses the user could say:\n"
        f"  1. A natural, on-topic reply\n"
        f"  2. A casual/fun response\n"
        f"  3. A question or curiosity-driven response\n"
        f"\n"
        f"OUTPUT FORMAT — strict JSON, no extra text:\n"
        f"{{\n"
        f"  \"response\": \"your immersive opening line\",\n"
        f"  \"suggested_replies\": [\"option 1\", \"option 2\", \"option 3\"]\n"
        f"}}"
    )
    
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Start the conversation for this scenario: {scenario}"}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.9,
    }
    
    BACKUP_MODELS = [
        "openai/gpt-oss-120b:free",
        "liquid/lfm-2.5-1.2b-instruct:free",
        "cohere/north-mini-code:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ]

    for model in [DEFAULT_MODEL] + BACKUP_MODELS:
        payload["model"] = model
        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(OPENROUTER_URL, json=payload, headers=get_headers())
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    parsed = json.loads(content)
                    if "response" in parsed and "suggested_replies" in parsed:
                        return parsed
                    logger.warning(f"Malformed start response from {model}")
                    continue
                elif response.status_code == 429:
                    logger.warning(f"Model {model} rate limited, trying next...")
                    continue
                else:
                    logger.error(f"OpenRouter start roleplay error ({model}): {response.status_code}")
                    continue
        except Exception as e:
            logger.error(f"Failed to start roleplay ({model}): {e}")
            continue
        
    return {
        "response": f"Hello! Let's practice speaking for our scenario: '{scenario}'. How are you doing today?",
        "suggested_replies": [
            "I'm doing well, thank you. Let's get started.",
            "Hey! I'm doing great, excited to practice this scenario.",
            "I'm good, thanks. What should we do first?"
        ]
    }

async def generate_custom_drills(contrast_prompt: str = None) -> dict:
    """Generates custom minimal pair word list using LLM."""
    if not contrast_prompt:
        contrast_prompt = "r vs l"
        
    prompt = (
        f"You are a speech therapist and phonetics coach.\n"
        f"Generate a minimal pairs drill contrast based on this prompt: '{contrast_prompt}'.\n"
        f"Provide a label for the contrast (e.g. '/r/ vs /l/'), a brief description, "
        f"and exactly 3 pairs of matching minimal contrast words.\n"
        f"Format your output strictly as a JSON object with these keys:\n"
        f"{{\n"
        f"  \"label\": \"contrast label\",\n"
        f"  \"description\": \"brief practice instructions\",\n"
        f"  \"pairs\": [\n"
        f"    {{\"word1\": \"wordA1\", \"word2\": \"wordA2\"}},\n"
        f"    {{\"word1\": \"wordB1\", \"word2\": \"wordB2\"}},\n"
        f"    {{\"word1\": \"wordC1\", \"word2\": \"wordC2\"}}\n"
        f"  ]\n"
        f"}}"
    )
    
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful speech therapist assistant."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(OPENROUTER_URL, json=payload, headers=get_headers())
            if response.status_code == 200:
                import json
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                return json.loads(content)
            else:
                logger.error(f"OpenRouter generate drills error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Failed to query OpenRouter drills: {e}")
        
    return {
        "label": "/r/ vs /l/",
        "description": "Distinguish between /r/ and /l/ liquid sounds.",
        "pairs": [
            {"word1": "read", "word2": "lead"},
            {"word1": "road", "word2": "load"},
            {"word1": "right", "word2": "light"}
        ]
    }

import re
import logging
import replicate
import anthropic
from openai import OpenAI
import google.generativeai as genai
import fal_client
from config import OPENROUTER_API_KEY, ANTHROPIC_API_KEY, REPLICATE_API_TOKEN, FAL_API_KEY, GEMINI_API_KEY, CHANNEL_INTRODUCTION, OPENAI_API_KEY

# Initialize clients safely
# OpenRouter uses the OpenAI SDK format with a different base URL
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
) if OPENROUTER_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
# Replicate automatically picks up REPLICATE_API_TOKEN from env vars

def split_transcript_into_chunks(text: str, max_chars: int = 10000) -> list[str]:
    """Split a transcript at sentence boundaries so each chunk is ≤ max_chars."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[.!?।。！？])\s+', text.strip())
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        slen = len(sentence)
        if current and current_len + slen + 1 > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [sentence], slen
        else:
            current.append(sentence)
            current_len += slen + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def translate_chunk(chunk: str, target_lang: str) -> str:
    """Translate / rewrite a single transcript chunk using the fastest available model.

    Uses claude-haiku-4-5 via OpenRouter (5× faster, 10× cheaper than Sonnet)
    for translation-only work; falls back to Sonnet/GPT-4o if Haiku isn't available.
    Returns plain translated text (no tags, no headers).
    """
    system_prompt = (
        f"You are a celebrated {target_lang} author who has spent your career writing historical audiobooks. "
        f"{target_lang} is your mother tongue. You think in {target_lang}. You dream in {target_lang}.\n\n"

        f"You have just read a source transcript in another language. "
        f"Now you will write your own {target_lang} book chapter about the same historical events — "
        f"from scratch, in your own voice, the way you would naturally tell this story to a {target_lang} reader.\n\n"

        f"THIS IS NOT A TRANSLATION. This is original {target_lang} authorship.\n\n"

        f"WHAT THAT MEANS IN PRACTICE:\n"
        f"- Read the source for facts and narrative. Then close it mentally and write freely.\n"
        f"- Every sentence must be constructed the way a native {target_lang} author constructs sentences — "
        f"not the way the source language constructs them.\n"
        f"- Use idioms, expressions, rhythms, and vocabulary that feel completely natural in {target_lang}. "
        f"If a phrase sounds like it was imported from another language, rewrite it entirely.\n"
        f"- Choose words a {target_lang} writer would reach for first, not the closest dictionary equivalent.\n\n"

        f"VOICE & STYLE:\n"
        f"- Warm, literary, and elegant — like a masterfully written historical audiobook.\n"
        f"- Calm and unhurried. Each paragraph should breathe and settle before moving forward.\n"
        f"- Intimate and immersive — the listener should feel they are being told a story by a trusted friend, "
        f"not lectured by a machine.\n"
        f"- Every sentence should flow naturally into the next. Read it aloud in your mind. "
        f"If it sounds awkward when spoken, rewrite it.\n\n"

        f"STRICTLY FORBIDDEN:\n"
        f"- Literal sentence-by-sentence translation.\n"
        f"- Carrying over the source language's sentence structure into {target_lang}.\n"
        f"- Unnatural connectors, mechanical transitions, or AI-sounding phrases.\n"
        f"- Repetitive patterns ('furthermore', 'moreover', 'it is worth noting', 'in conclusion', etc.).\n"
        f"- Any word or phrase that a native {target_lang} author would not choose naturally.\n"
        f"- Advertisements, sponsorships, channel promotions, or references to other creators.\n\n"

        f"FACTS ARE SACRED, WORDS ARE FREE:\n"
        f"- Preserve every historical fact exactly: names, dates, places, events, figures.\n"
        f"- But rewrite every sentence around those facts in pure {target_lang}.\n"
        f"- Convert all numerals to full words in {target_lang}.\n\n"

        f"OUTPUT: A single continuous block of flowing {target_lang} prose. "
        f"No headers, no tags, no commentary. Just the story."
    )
    try:
        if openrouter_client:
            # Use Haiku for speed on translation-only chunks
            resp = openrouter_client.chat.completions.create(
                model="anthropic/claude-haiku-4-5",
                max_tokens=8192,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk},
                ],
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        elif anthropic_client:
            msg = anthropic_client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=8192,
                system=system_prompt,
                messages=[{"role": "user", "content": chunk}],
                temperature=0.7,
            )
            return msg.content[0].text.strip()
    except Exception as e:
        logging.error(f"translate_chunk error: {e}")
        raise   # re-raise so the retry wrapper in bot.py can catch it
    return chunk   # fallback


def split_into_paragraphs(text: str, min_chars: int = 500, max_chars: int = 999) -> list[str]:
    """Splits text into paragraphs where each paragraph is under max_chars
    and usually between min_chars and max_chars. Sentence boundary checks support
    both Western and CJK punctuation.
    """
    sentences = re.split(r'(?<=[.!?。！？])\s*', text.strip())
    paragraphs = []
    current_para = []
    current_len = 0
    
    for sentence in sentences:
        if not sentence:
            continue
        sentence_len = len(sentence)
        if current_para and (current_len + 1 + sentence_len > max_chars):
            paragraphs.append(" ".join(current_para))
            current_para = [sentence]
            current_len = sentence_len
        else:
            if current_para:
                current_para.append(sentence)
                current_len += 1 + sentence_len
            else:
                current_para = [sentence]
                current_len = sentence_len
            
            if current_len >= min_chars:
                paragraphs.append(" ".join(current_para))
                current_para = []
                current_len = 0
                
    if current_para:
        paragraphs.append(" ".join(current_para))
        
    return paragraphs

def generate_text_and_extract_prompt(transcript: str, system_prompt: str, target_lang: str = "English") -> tuple[str, str, str]:
    """Calls Claude (primary), Gemini, or OpenAI to translate, rewrite, extract thumbnail prompt,
    and format the transcript into paragraphs with the channel's custom introduction.
    """
    if "Production Pack" in system_prompt or "premium media studio" in system_prompt:
        lang_instruction = (
            f"\n\nCRITICAL LANGUAGE AND SCRIPT REWRITE INSTRUCTION:\n"
            f"The target language for this request is {target_lang}.\n"
            f"OUTPUT ORDER IS MANDATORY — follow this sequence exactly:\n\n"
            f"STEP A — Output these XML blocks FIRST, before anything else:\n"
            f"  A1. Literally and professionally translate the following channel introduction into {target_lang} "
            f"and wrap it in <translated_intro>...</translated_intro> tags:\n"
            f"  \"{CHANNEL_INTRODUCTION}\"\n\n"
            f"  A2. Completely rebuild and rewrite the input transcript into a premium historical documentary script "
            f"in {target_lang} designed for relaxation and sleep. Wrap it in <translated_transcript>...</translated_transcript> tags.\n"
            f"  Rules for the rewritten script:\n"
            f"   - Never translate literally — extract the ideas, historical facts, and structure, then write a completely new, better script.\n"
            f"   - Keep the narrative logical, calm, and soothing; maintain historical facts (names, cities, dates) but rewrite all phrasing.\n"
            f"   - Convert all numerals into full words in {target_lang}.\n"
            f"   - One single continuous block of text inside the tags, no paragraph breaks or comments.\n"
            f"   - Remove all advertisements, sponsored content, promotions, and mentions of other channels.\n\n"
            f"STEP B — After the XML blocks, output the Production Pack exactly as structured:\n"
            f"  Use {target_lang} for all target-language content and also provide Russian translations (labeled 🇷🇺:).\n"
            f"  Do NOT translate the AI host photo prompt or thumbnail prompt — keep them in English.\n"
            f"  Wrap the thumbnail prompt in <thumbnail_prompt>...</thumbnail_prompt> inside Section 8."
        )
    else:
        lang_instruction = (
            f"\n\nCRITICAL INSTRUCTION — OUTPUT ORDER IS MANDATORY:\n\n"
            f"STEP A — Output these XML blocks FIRST, before anything else:\n"
            f"  A1. Literally and professionally translate the following channel introduction into {target_lang} "
            f"and wrap it in <translated_intro>...</translated_intro> tags (faithful translation, no rewriting):\n"
            f"  \"{CHANNEL_INTRODUCTION}\"\n\n"
            f"  A2. Translate, revise, and clean the transcript into {target_lang}. "
            f"Wrap it in <translated_transcript>...</translated_transcript> tags.\n"
            f"  Rules:\n"
            f"  - Make subtitles look normal and natural.\n"
            f"  - Significantly revise phrasing — make it a completely new script while preserving meaning.\n"
            f"  - Keep the narrative logical; you may rearrange semantic blocks but preserve narrative logic.\n"
            f"  - Do NOT change names, titles, cities, countries, dates, times, or factual details.\n"
            f"  - Correct grammar and punctuation per {target_lang} rules.\n"
            f"  - Convert all numerals into full words in {target_lang}.\n"
            f"  - One single continuous block of text inside the tags, no paragraph breaks or extra comments.\n"
            f"  - Remove all advertisements, sponsorships, promotions, and mentions of other channels.\n\n"
            f"STEP B — After the XML blocks, generate the YouTube titles, description, hashtags, and tags in {target_lang}."
        )

    full_text = ""

    if openrouter_client:
        response = openrouter_client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            max_tokens=8192,
            messages=[
                {"role": "system", "content": system_prompt + lang_instruction},
                {"role": "user", "content": f"Here is the transcript of the video:\n\n{transcript}"}
            ],
            temperature=0.7,
        )
        full_text = response.choices[0].message.content
        logging.info(f"OpenRouter response length: {len(full_text)} chars, preview: {full_text[:100]!r}")
    elif anthropic_client:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8096,
            system=system_prompt + lang_instruction,
            messages=[
                {"role": "user", "content": f"Here is the transcript of the video:\n\n{transcript}"}
            ],
            temperature=0.7,
        )
        full_text = message.content[0].text
    elif GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=system_prompt + lang_instruction
        )
        response = model.generate_content(
            f"Here is the transcript of the video:\n\n{transcript}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
            )
        )
        full_text = response.text
    else:
        if "Production Pack" in system_prompt or "premium media studio" in system_prompt:
            mock_prompt = "A cinematic Netflix-style documentary portrait of a Roman legionary standing in the rain, ultra realistic, cinematic lighting"
            mock_intro = f"[MOCK INTRO IN {target_lang.upper()}] I invite you to a new bedtime story for relaxation and sleep. Get comfortable, relax, and let yourself be carried away on this journey..."
            mock_translation = f"[MOCK REWRITTEN TRANSCRIPT IN {target_lang.upper()}] Once upon a time, Rome was a small village, but over the centuries it grew into one of the greatest empires in human history. Let yourself relax as we explore this fascinating journey..."
            full_text = f"""<translated_intro>{mock_intro}</translated_intro>
<translated_transcript>{mock_translation}</translated_transcript>
# 1. TITLE IDEAS
{target_lang}: The Rise of the Roman Empire
🇷🇺: Восхождение Римской Империи

{target_lang}: Echoes of the Caesars
🇷🇺: Отголоски Цезарей

{target_lang}: Shadows of Rome: A Relaxing Documentary
🇷🇺: Тени Рима: Расслабляющий Документальный Фильм

--------------------------------------------------
# 2. BEST TITLE
{target_lang}: Shadows of Rome: A Relaxing Documentary
🇷🇺: Тени Рима: Расслабляющий Документальный Фильм

--------------------------------------------------
# 3. SEO DESCRIPTION
{target_lang}: Join us on a relaxing journey through the historic streets of ancient Rome. Perfect for sleep, study, or deep relaxation.
🇷🇺: Присоединяйтесь к нам в расслабляющем путешествии по историческим улицам древнего Рима. Идеально для сна, учебы или глубокого расслабления.

--------------------------------------------------
# 4. HASHTAGS
#Rome #History #Relaxation #SleepDocumentary

--------------------------------------------------
# 5. TAGS
ancient rome, history documentary, relaxation sleep, roman empire, bedtime story

--------------------------------------------------
# 6. AI HOST SCRIPT
{target_lang}: Welcome back. Tonight, we journey into the heart of ancient Rome. Relax, close your eyes, and enjoy the story.
🇷🇺: С возвращением. Сегодня мы отправимся в самое сердце древнего Рима. Расслабьтесь, закройте глаза и наслаждайтесь историей.

--------------------------------------------------
# 7. AI HOST PHOTO PROMPT
A close-up studio portrait of a 45-year-old male historian with short grey hair and a calm expression, wearing a tailored dark wool trenchcoat in a library, warm soft lighting, Netflix documentary style.

--------------------------------------------------
# 8. THUMBNAIL PROMPT
<thumbnail_prompt>{mock_prompt}</thumbnail_prompt>"""
        else:
            mock_text = f"*(MOCK MODE - {target_lang})*\n\n**Titles:**\n1. Dummy Title 1 in {target_lang}\n2. Dummy Title 2 in {target_lang}\n3. Dummy Title 3 in {target_lang}\n\n**Description:**\nThis is a dummy description in {target_lang}.\n\n**Tags:** #mock #test"
            mock_prompt = "A fake thumbnail prompt"
            mock_translation = f"[MOCK TRANSCRIPT IN {target_lang.upper()}] This is a mock translated transcript."
            full_text = f"<translated_intro>[MOCK INTRO IN {target_lang.upper()}] I invite you to a new bedtime story for relaxation and sleep. Get comfortable, relax, and let yourself be carried away on this journey...</translated_intro>\n<translated_transcript>{mock_translation}</translated_transcript>\n{mock_text}\n<thumbnail_prompt>{mock_prompt}</thumbnail_prompt>"

    # Extract the translated intro
    translated_intro = ""
    match_intro = re.search(r'<translated_intro>(.*?)</translated_intro>', full_text, re.DOTALL)
    if match_intro:
        translated_intro = match_intro.group(1).strip()
        full_text = re.sub(r'<translated_intro>.*?</translated_intro>', '', full_text, flags=re.DOTALL).strip()

    # Extract the translated transcript
    translated_transcript = ""
    match_trans = re.search(r'<translated_transcript>(.*?)</translated_transcript>', full_text, re.DOTALL)
    if match_trans:
        translated_transcript = match_trans.group(1).strip()
        full_text = re.sub(r'<translated_transcript>.*?</translated_transcript>', '', full_text, flags=re.DOTALL).strip()

    # Extract the thumbnail prompt
    thumbnail_prompt = "A generic YouTube thumbnail"
    match_thumb = re.search(r'<thumbnail_prompt>(.*?)</thumbnail_prompt>', full_text, re.DOTALL)
    if match_thumb:
        thumbnail_prompt = match_thumb.group(1).strip()
        full_text = full_text.replace("<thumbnail_prompt>", "").replace("</thumbnail_prompt>", "").strip()

    # Form the final transcript: split the main body, then prepend the intro paragraph
    final_transcript = ""
    if translated_transcript:
        paragraphs = split_into_paragraphs(translated_transcript, min_chars=500, max_chars=999)
        if translated_intro:
            paragraphs.insert(0, translated_intro)
        final_transcript = "\n\n".join(paragraphs)

    return full_text, thumbnail_prompt, final_transcript

def generate_thumbnail(image_prompt: str) -> str:
    """Generate a FLUX 1.1 Pro thumbnail. Uses fal.ai if key is set, falls back to Replicate."""

    # ── fal.ai (primary) ────────────────────────────────────────────────────
    if FAL_API_KEY:
        try:
            import os
            os.environ["FAL_KEY"] = FAL_API_KEY
            result = fal_client.subscribe(
                "fal-ai/flux-pro/v1.1",
                arguments={
                    "prompt": image_prompt,
                    "aspect_ratio": "16:9",
                    "output_format": "jpeg",
                    "output_quality": 100,
                    "safety_tolerance": 6,
                },
            )
            return result["images"][0]["url"]
        except Exception as e:
            logging.error(f"fal.ai thumbnail error: {e}")

    # ── Replicate (fallback) ─────────────────────────────────────────────────
    if REPLICATE_API_TOKEN:
        try:
            output = replicate.run(
                "black-forest-labs/flux-1.1-pro",
                input={"prompt": image_prompt, "aspect_ratio": "16:9", "output_format": "jpg"},
            )
            result = output[0] if isinstance(output, list) else output
            return str(result.url) if hasattr(result, "url") else str(result)
        except Exception as e:
            logging.error(f"Replicate thumbnail error: {e}")

    return ""


def generate_intro_video(intro_text: str, target_lang: str) -> str:
    """Generate a talking-head intro video.

    Pipeline:
      1. Pick the culturally-styled host photo for the target language.
      2. Generate TTS audio from intro_text using OpenAI (onyx voice).
      3. Upload both files to fal.ai storage.
      4. Run fal-ai/sync-lipsync → returns a lip-synced video URL.
    """
    import os
    import tempfile

    # ── Language → host photo ────────────────────────────────────────────────
    LANG_CODE = {
        "English": "en", "Spanish": "es", "French": "fr", "German": "de",
        "Portuguese": "pt", "Italian": "it", "Chinese": "zh", "Japanese": "ja",
        "Russian": "ru", "Polish": "pl", "Romanian": "ro", "Turkish": "tr",
    }
    lang_code = LANG_CODE.get(target_lang, "en")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    photo_path = os.path.join(base_dir, "host_photos", f"{lang_code}.jpg")
    if not os.path.exists(photo_path):
        photo_path = os.path.join(base_dir, "host_photos", "en.jpg")

    if not os.path.exists(photo_path):
        logging.error(f"Host photo not found for lang '{lang_code}'")
        return ""

    # ── Step 1: TTS via gTTS (free, no API key required) ────────────────────
    GTTS_LANG = {
        "English": "en", "Spanish": "es", "French": "fr", "German": "de",
        "Portuguese": "pt", "Italian": "it", "Chinese": "zh-CN", "Japanese": "ja",
        "Russian": "ru", "Polish": "pl", "Romanian": "ro", "Turkish": "tr",
    }

    audio_path = None
    try:
        from gtts import gTTS
        gtts_lang = GTTS_LANG.get(target_lang, "en")
        tts = gTTS(text=intro_text, lang=gtts_lang, slow=False)
        audio_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(audio_tmp.name)
        audio_path = audio_tmp.name
        logging.info(f"TTS (gTTS/{gtts_lang}) generated: {audio_path}")
    except Exception as e:
        logging.error(f"gTTS failed: {e}")
        return ""

    # ── Step 2: Upload to fal storage & run lip-sync ─────────────────────────
    if not FAL_API_KEY:
        logging.error("FAL_API_KEY not set — cannot run lip-sync")
        if audio_path:
            os.unlink(audio_path)
        return ""

    try:
        os.environ["FAL_KEY"] = FAL_API_KEY

        photo_url = fal_client.upload_file(photo_path)
        audio_url = fal_client.upload_file(audio_path)
        logging.info(f"Uploaded photo → {photo_url[:60]}")
        logging.info(f"Uploaded audio → {audio_url[:60]}")

        result = fal_client.subscribe(
            "fal-ai/sadtalker",
            arguments={
                "source_image_url": photo_url,
                "driven_audio_url": audio_url,
            },
        )
        video_url = (
            result.get("video", {}).get("url", "")
            or result.get("video_url", "")
            or ""
        )
        logging.info(f"Lip-sync video URL: {video_url[:80] if video_url else 'EMPTY'}")
        return video_url
    except Exception as e:
        logging.error(f"fal.ai lip-sync error: {e}")
        return ""
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

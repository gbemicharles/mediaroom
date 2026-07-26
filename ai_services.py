import re
import logging
import replicate
import anthropic
from openai import OpenAI
import google.generativeai as genai
import fal_client
from config import OPENROUTER_API_KEY, ANTHROPIC_API_KEY, REPLICATE_API_TOKEN, FAL_API_KEY, GEMINI_API_KEY, CHANNEL_INTRODUCTION, OPENAI_API_KEY, HEYGEN_API_KEY

# Initialize clients safely
# OpenRouter uses the OpenAI SDK format with a different base URL
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
) if OPENROUTER_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
# Replicate automatically picks up REPLICATE_API_TOKEN from env vars

# Physical appearance descriptions for each avatar character.
# Used to ground Section 6 (AI HOST PHOTO PROMPT) so the LLM dresses
# the correct person in period-appropriate clothing, never modern.
AVATAR_APPEARANCE = {
    "English":    "Northern European male, approximately 38 years old, short dark brown hair, clean-shaven, calm blue-grey eyes, strong jaw",
    "Spanish":    "Mediterranean male, approximately 40 years old, dark wavy hair, warm olive skin, expressive dark eyes, short beard",
    "French":     "Western European male, approximately 35 years old, medium-length styled dark hair, refined angular features, confident bearing",
    "German":     "Germanic male, approximately 36 years old, short blonde-brown hair, athletic build, sharp defined features, clean-shaven",
    "Portuguese": "Mediterranean male, approximately 40 years old, dark wavy hair, warm olive skin, expressive dark eyes, short beard",
    "Italian":    "Western European male, approximately 35 years old, medium-length styled dark hair, refined angular features, confident bearing",
    "Chinese":    "East Asian male, approximately 32 years old, neat short black hair, refined scholarly appearance, calm dark eyes",
    "Japanese":   "East Asian male, approximately 32 years old, neat short black hair, refined scholarly appearance, calm dark eyes",
    "Russian":    "Eastern European male, approximately 45 years old, salt-and-pepper short hair, distinguished mature features, strong weathered presence",
    "Polish":     "Eastern European male, approximately 45 years old, salt-and-pepper short hair, distinguished mature features, strong weathered presence",
    "Romanian":   "Eastern European male, approximately 45 years old, salt-and-pepper short hair, distinguished mature features, strong weathered presence",
    "Turkish":    "Mixed heritage male, approximately 33 years old, short textured dark hair, warm olive-brown skin, open friendly face, light stubble",
}


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
        # Look up the avatar appearance for this language (fall back to English)
        _avatar_appearance = AVATAR_APPEARANCE.get(target_lang, AVATAR_APPEARANCE["English"])
        lang_instruction = (
            f"\n\nCRITICAL LANGUAGE AND SCRIPT REWRITE INSTRUCTION:\n"
            f"The target language for this request is {target_lang}.\n\n"
            f"AI HOST CHARACTER — the permanent host for {target_lang} content is: {_avatar_appearance}.\n"
            f"Use this description as the subject in Section 6 (AI HOST PHOTO PROMPT).\n\n"
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

def _pick_font_path(text: str) -> str:
    """Return best available font path for the given text's script."""
    import os
    BASE = os.path.dirname(os.path.abspath(__file__))

    def _has(path): return os.path.exists(path)

    # Detect script by Unicode range
    has_cjk      = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in text)
    has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in text)

    cjk_font  = os.path.join(BASE, "fonts", "NotoSansCJK.otf")
    noto_font = os.path.join(BASE, "fonts", "NotoSans.ttf")
    bebas     = os.path.join(BASE, "fonts", "BebasNeue.ttf")
    dejavu    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    if has_cjk and _has(cjk_font):
        return cjk_font
    if has_cyrillic and _has(noto_font):
        return noto_font
    if _has(bebas):
        return bebas          # Bebas Neue — condensed, cinematic, all-caps
    if _has(noto_font):
        return noto_font
    return dejavu             # system fallback


def _burn_hook_text(image_url: str, hook_text: str) -> str:
    """Download thumbnail, burn cinematic gold title text with PIL, re-upload.

    Style matches premium YouTube thumbnails:
    • Bebas Neue / Noto font (script-aware)
    • Gold fill (#FFD700) with deep drop shadow — no dark bar
    • Large, upper-center placement
    """
    import os, tempfile, requests
    from PIL import Image, ImageDraw, ImageFont

    TARGET_WIDTH_RATIO = 0.86   # text spans up to 86 % of width

    def _best_wrap(draw, text: str, W: int, H: int, font_path: str):
        words = text.split()
        best  = (None, None, 0)          # (lines, font, font_size)
        candidates = [" ".join(words)]
        for split in range(1, len(words)):
            candidates.append(" ".join(words[:split]) + "\n" + " ".join(words[split:]))
        for candidate in candidates:
            lines = candidate.split("\n")
            if len(lines) > 2:
                continue
            # Allow generous size: H//2 for 1 line, H//3 per line for 2 lines
            max_size = H // (2 if len(lines) == 1 else 3)
            lo, hi, chosen_size = 10, max_size, 10
            while lo <= hi:
                mid = (lo + hi) // 2
                f   = ImageFont.truetype(font_path, mid)
                max_w = max(draw.textbbox((0, 0), l, font=f)[2] for l in lines)
                if max_w <= W * TARGET_WIDTH_RATIO:
                    chosen_size = mid; lo = mid + 1
                else:
                    hi = mid - 1
            if chosen_size > best[2]:
                best = (lines, ImageFont.truetype(font_path, chosen_size), chosen_size)
        return best[0], best[1], best[2]

    try:
        # Retry up to 3 times — FAL CDN can be slow to serve a freshly-generated file
        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(image_url, timeout=60)
                resp.raise_for_status()
                break
            except Exception as dl_err:
                if attempt < 2:
                    import time
                    logging.warning(f"_burn_hook_text download attempt {attempt+1} failed: {dl_err} — retrying in 5s")
                    time.sleep(5)
                else:
                    raise

        img_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img_tmp.write(resp.content)
        img_tmp.close()

        img  = Image.open(img_tmp.name).convert("RGB")
        draw = ImageDraw.Draw(img)
        W, H = img.size

        text      = hook_text.upper().strip()
        font_path = _pick_font_path(text)
        lines, font, font_size = _best_wrap(draw, text, W, H, font_path)
        if not lines or not font:
            logging.warning("_burn_hook_text: could not fit text, returning original")
            return image_url

        # Measure block
        line_bboxes  = [draw.textbbox((0, 0), l, font=font) for l in lines]
        line_heights = [bb[3] - bb[1] for bb in line_bboxes]
        line_widths  = [bb[2] - bb[0] for bb in line_bboxes]
        gap          = max(6, font_size // 10)
        block_h      = sum(line_heights) + gap * (len(lines) - 1)

        # ── Position: upper-centre (15 % from top) ───────────────────────────
        y_start       = int(H * 0.12)
        shadow_offset = max(5, font_size // 16)
        glow_r        = max(2, font_size // 35)

        GOLD        = (255, 215,   0)   # #FFD700
        DARK_AMBER  = (140,  80,   0)   # warm glow halo
        BLACK       = (  0,   0,   0)

        for i, line in enumerate(lines):
            x = (W - line_widths[i]) // 2
            y = y_start + sum(line_heights[:i]) + gap * i

            # 1. Deep drop shadow (stacked, bottom-right)
            for s in range(shadow_offset, 0, -1):
                draw.text((x + s, y + s), line, font=font, fill=BLACK)

            # 2. Warm amber glow halo around the text
            for dx in range(-glow_r, glow_r + 1):
                for dy in range(-glow_r, glow_r + 1):
                    if dx or dy:
                        draw.text((x + dx, y + dy), line, font=font, fill=DARK_AMBER)

            # 3. Gold fill on top
            draw.text((x, y), line, font=font, fill=GOLD)

        out_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(out_tmp.name, "JPEG", quality=95)
        out_tmp.close()

        os.environ["FAL_KEY"] = FAL_API_KEY
        new_url = fal_client.upload_file(out_tmp.name)

        os.unlink(img_tmp.name)
        os.unlink(out_tmp.name)
        font_name = os.path.basename(font_path)
        logging.info(f"Title '{text[:60]}' burned ({font_size}px {font_name}, gold, {len(lines)} line(s))")
        return new_url
    except Exception as e:
        logging.error(f"PIL title overlay failed: {e}")
        return image_url


def generate_thumbnail(image_prompt: str, hook_text: str = "") -> str:
    """Generate a FLUX 1.1 Pro thumbnail and burn hook text on top with PIL."""
    import os
    os.environ.setdefault("FAL_KEY", FAL_API_KEY or "")

    # Use the image prompt as-is — do NOT ask FLUX to render text.
    # PIL burns the hook text on top cleanly in a second pass.
    # Injecting text into the FLUX prompt causes FLUX to render its own
    # version which clashes with (and shows through) the PIL overlay.
    flux_prompt = image_prompt

    image_url = ""

    # ── fal.ai (primary) ────────────────────────────────────────────────────
    if FAL_API_KEY:
        try:
            os.environ["FAL_KEY"] = FAL_API_KEY
            result = fal_client.subscribe(
                "fal-ai/flux-pro/v1.1",
                arguments={
                    "prompt": flux_prompt,
                    "image_size": "landscape_16_9",
                    "output_format": "jpeg",
                    "output_quality": 100,
                    "safety_tolerance": 6,
                },
            )
            image_url = result["images"][0]["url"]
        except Exception as e:
            logging.error(f"fal.ai thumbnail error: {e}")

    # ── Replicate (fallback) ─────────────────────────────────────────────────
    if not image_url and REPLICATE_API_TOKEN:
        try:
            output = replicate.run(
                "black-forest-labs/flux-1.1-pro",
                input={"prompt": image_prompt, "aspect_ratio": "16:9", "output_format": "jpg"},
            )
            result = output[0] if isinstance(output, list) else output
            image_url = str(result.url) if hasattr(result, "url") else str(result)
        except Exception as e:
            logging.error(f"Replicate thumbnail error: {e}")

    # ── Burn hook text on top with PIL ───────────────────────────────────────
    if image_url and hook_text and FAL_API_KEY:
        image_url = _burn_hook_text(image_url, hook_text)

    return image_url


def regenerate_section(section_num: int, transcript: str, target_lang: str,
                       full_pack_text: str, system_prompt: str) -> str:
    """Re-generate a single production-pack section independently.

    Returns the raw LLM text for that section (including its markdown header).
    Returns empty string on failure.
    """
    section_names = {
        1: "TITLE IDEAS",
        2: "WINNER",
        3: "SEO DESCRIPTION & HASHTAGS",
        4: "TAGS",
        5: "AI HOST SCRIPT",
        6: "AI HOST PHOTO PROMPT",
        7: "THUMBNAIL PROMPT",
    }
    name = section_names.get(section_num, f"Section {section_num}")

    tag_note = (
        "\nWrap the image generation prompt in <thumbnail_prompt>...</thumbnail_prompt> tags, "
        "exactly as in the original format."
        if section_num == 7 else ""
    )

    regen_system = (
        f"{system_prompt}\n\n"
        f"━━━ REGENERATION INSTRUCTION ━━━\n"
        f"Output ONLY Section {section_num}: {name}. "
        f"Do NOT output any other section — not even their headers.\n"
        f"Target language: {target_lang}.\n"
        f"Make this version noticeably different from the existing one — "
        f"try a different angle, structure, or framing while being equally strong or stronger."
        f"{tag_note}"
    )
    user_content = (
        f"TRANSCRIPT (target-language version):\n{transcript[:4000]}\n\n"
        f"EXISTING PRODUCTION PACK (for context — do NOT repeat anything from it verbatim):\n"
        f"{full_pack_text[:2000]}\n\n"
        f"Now regenerate ONLY Section {section_num}: {name}."
    )

    result = ""
    try:
        if openrouter_client:
            resp = openrouter_client.chat.completions.create(
                model="anthropic/claude-sonnet-4-5",
                max_tokens=1500,
                temperature=0.85,
                messages=[
                    {"role": "system", "content": regen_system},
                    {"role": "user",   "content": user_content},
                ],
            )
            result = resp.choices[0].message.content.strip()
        elif anthropic_client:
            resp = anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1500,
                temperature=0.85,
                system=regen_system,
                messages=[{"role": "user", "content": user_content}],
            )
            result = resp.content[0].text.strip()
    except Exception as e:
        logging.error(f"regenerate_section({section_num}) error: {e}")

    return result


def _detect_story_theme(text: str) -> str:
    """Classify story theme from keywords. Returns one of: bedtime, adventure, nature, educational, general."""
    low = text.lower()
    bedtime_kw   = {"sleep", "bedtime", "night", "dream", "relax", "story", "cozy", "peaceful",
                    "rest", "slumber", "calm", "soothing", "lullaby", "gentle", "tonight", "drift",
                    "unwind", "sleepy", "goodnight", "tales", "comfort", "snuggle"}
    adventure_kw = {"adventure", "journey", "mystery", "explore", "quest", "expedition",
                    "discover", "travel", "wander", "mission", "escape", "hero", "battle"}
    nature_kw    = {"nature", "forest", "ocean", "mountain", "wildlife", "earth", "garden",
                    "river", "lake", "jungle", "wilderness", "desert", "sea", "trees", "birds"}
    edu_kw       = {"learn", "education", "guide", "explain", "science", "history", "facts",
                    "understand", "knowledge", "discover", "research", "study", "how", "why"}

    words = set(low.split())
    scores = {
        "bedtime":     len(words & bedtime_kw),
        "adventure":   len(words & adventure_kw),
        "nature":      len(words & nature_kw),
        "educational": len(words & edu_kw),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else "general"


def _generate_background_image(prompt: str) -> str:
    """Generate a thematic background image via FLUX schnell (fast, cheap).

    Uses the story/thumbnail prompt but strips text elements and people.
    Returns a CDN URL or empty string on failure.
    """
    import os
    if not FAL_API_KEY:
        return ""
    os.environ["FAL_KEY"] = FAL_API_KEY
    bg_prompt = (
        f"{prompt.rstrip('. ')}. "
        "Cinematic background scene, no people, no faces, no text, no watermarks, "
        "atmospheric lighting, high production value, suitable as a video studio background, "
        "soft depth of field, ultra-detailed environment."
    )
    try:
        result = fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": bg_prompt,
                "image_size": "landscape_16_9",
                "num_inference_steps": 4,
                "num_images": 1,
            },
        )
        url = result["images"][0]["url"]
        logging.info(f"Background image generated: {url[:80]}")
        return url
    except Exception as e:
        logging.warning(f"Background image generation failed (non-fatal): {e}")
        return ""


def generate_intro_video(intro_text: str, target_lang: str, image_prompt: str = "") -> str:
    """Generate a ~10-second full-body intro video via HeyGen v3.

    Pipeline:
      1. Detect story theme → pick thematic avatar variant + generate background.
      2. LLM condenses intro_text to ~20-28 words in target_lang.
      3. Submit HeyGen avatar video job with custom background image.
      4. Poll until complete (max 3 min) → return video URL.
    """
    import time
    import requests

    # ── Culturally matched full-body standing male avatars ────────────────────
    # Only Avatar-V-compatible IDs work with script+voice_id on /v3/videos.
    # Jin/Onat/Marcus/Jinwoo/Aditya/Ivan are Avatar IV only and return 400.
    # Variants listed here are confirmed to return 200 from prior live tests.
    #
    # "bedtime" theme → sofa/casual variants (warmer, more intimate feel)
    # "adventure/nature" theme → outdoor/gym variants
    # "educational/general" → office/training variants (default)
    # Physical appearance descriptions for each avatar character.
    # Used to ground Section 7 (AI HOST PHOTO PROMPT) so the LLM can
    # dress the correct person in period-appropriate clothing.
    AVATAR_APPEARANCE = {
        "English":    "Northern European male, approximately 38 years old, short dark brown hair, clean-shaven, calm blue-grey eyes, strong jaw",
        "Spanish":    "Mediterranean male, approximately 40 years old, dark wavy hair, warm olive skin, expressive dark eyes, short beard",
        "French":     "Western European male, approximately 35 years old, medium-length styled dark hair, refined angular features, confident bearing",
        "German":     "Germanic male, approximately 36 years old, short blonde-brown hair, athletic build, sharp defined features, clean-shaven",
        "Portuguese": "Mediterranean male, approximately 40 years old, dark wavy hair, warm olive skin, expressive dark eyes, short beard",
        "Italian":    "Western European male, approximately 35 years old, medium-length styled dark hair, refined angular features, confident bearing",
        "Chinese":    "East Asian male, approximately 32 years old, neat short black hair, refined scholarly appearance, calm dark eyes",
        "Japanese":   "East Asian male, approximately 32 years old, neat short black hair, refined scholarly appearance, calm dark eyes",
        "Russian":    "Eastern European male, approximately 45 years old, salt-and-pepper short hair, distinguished mature features, strong weathered presence",
        "Polish":     "Eastern European male, approximately 45 years old, salt-and-pepper short hair, distinguished mature features, strong weathered presence",
        "Romanian":   "Eastern European male, approximately 45 years old, salt-and-pepper short hair, distinguished mature features, strong weathered presence",
        "Turkish":    "Mixed heritage male, approximately 33 years old, short textured dark hair, warm olive-brown skin, open friendly face, light stubble",
    }

    AVATAR_ID = {
        # ── defaults (educational / general) ─────────────────────────────────
        "English":    "Noah_standing_office_front",
        "Spanish":    "Raul_standing_office_front",
        "French":     "Vince_standing_businesstraining_front",
        "German":     "Jonas_standing_gym_front",
        "Portuguese": "Raul_standing_office_front",
        "Italian":    "Vince_standing_sofacasual_front",
        "Chinese":    "Ren_standing_office_front",
        "Japanese":   "Ren_standing_office_front",
        "Russian":    "Teodor_standing_office_front",
        "Polish":     "Teodor_standing_office_front",
        "Romanian":   "Teodor_standing_office_front",
        "Turkish":    "Miles_standing_outdoor_front",
    }

    # Theme overrides — only confirmed-working variants are listed here.
    # Variants not listed fall back to the default above.
    AVATAR_THEME_OVERRIDE = {
        "bedtime": {
            "English": "Noah_standing_sofa_front",          # confirmed 200
            "French":  "Vince_standing_sofacasual_front",   # confirmed 200
            "Italian": "Vince_standing_sofacasual_front",   # confirmed 200
            "Turkish": "Miles_standing_sofa_front",         # confirmed 200
        },
        "adventure": {
            "Turkish": "Miles_standing_outdoor_front",      # already outdoor
        },
        "nature": {
            "Turkish": "Miles_standing_outdoor_front",
        },
    }

    # ── Natural male HeyGen voices per language ───────────────────────────────
    VOICE_ID = {
        "English":    "828b59f834fd4c7188da322b6d9b6c75",  # David Castlemore — warm, authoritative
        "Spanish":    "626ca51acb2e496f8dcee8d7591fda3c",  # Narrator Mateo — storytelling
        "French":     "25a6a67280574d3da78e97b1935ebfc7",  # Émile Noir — cinematic French
        "German":     "5bc2783d1d1b45f0973a1ce8e269c35d",  # Vibrant Viktor — energetic German
        "Portuguese": "c8ac31e97555494fb8502599e6bc5461",  # Adriano — natural Brazilian/Portuguese
        "Italian":    "23afa694410d427d9ec632080079b74f",  # Voce Minatore Audiolibro — audiobook
        "Chinese":    "735c507fdc844be3b1528dd33f7dfb2a",  # Martin Li — professional Chinese
        "Japanese":   "662e1397965c484e8f65fa58c77effde",  # Satoshi — natural Japanese male
        "Russian":    "5f99970adadb42398bf1aeb963a3888b",  # Dmitry — deep Russian
        "Polish":     "f2d44cfdd1dc4846ae54a01d9db9d9fe",  # Marek - Natural — Polish
        "Romanian":   "ec218e50cc9c4991894676a31e4804c5",  # Emil - Natural — Romanian
        "Turkish":    "836aa05e398543d08231f68bffdfc025",  # Deniz Yılmaz — Turkish
    }

    # ── Theme detection → pick avatar variant ────────────────────────────────
    theme = _detect_story_theme(intro_text)
    logging.info(f"Detected story theme: {theme!r}")
    theme_overrides = AVATAR_THEME_OVERRIDE.get(theme, {})
    avatar_id = theme_overrides.get(target_lang) or AVATAR_ID.get(target_lang, AVATAR_ID["English"])
    voice_id  = VOICE_ID.get(target_lang, VOICE_ID["English"])
    logging.info(f"Avatar: {avatar_id}  Voice: {voice_id}")

    # ── Background image ──────────────────────────────────────────────────────
    # Generate a thematic scene image from the thumbnail/story prompt (cheap FLUX schnell).
    # Falls back gracefully — no background sent to HeyGen if generation fails.
    bg_url = ""
    if image_prompt:
        logging.info("Generating thematic background image for intro video…")
        bg_url = _generate_background_image(image_prompt)

    if not HEYGEN_API_KEY:
        logging.error("HEYGEN_API_KEY not set")
        return ""

    HEYGEN_HEADERS = {"X-Api-Key": HEYGEN_API_KEY, "Content-Type": "application/json"}

    # ── Step 1: LLM → concise 10-second script in target language ────────────
    condensed_script = ""
    try:
        system_prompt = (
            f"You are a warm, storytelling YouTube host writing a 10-second spoken intro "
            f"in {target_lang}. Capture the spirit of the reference text: invite the viewer "
            f"to relax, hint at tonight's story, and wish them peaceful listening. "
            f"Write exactly 20–28 words — no more. Natural, flowing, conversational speech. "
            f"Output ONLY the final script — no quotes, no labels, nothing else."
        )
        if openrouter_client:
            resp = openrouter_client.chat.completions.create(
                model="google/gemini-2.5-flash",
                max_tokens=120,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": intro_text},
                ],
            )
            condensed_script = resp.choices[0].message.content.strip()
        elif anthropic_client:
            resp = anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=120,
                system=system_prompt,
                messages=[{"role": "user", "content": intro_text}],
            )
            condensed_script = resp.content[0].text.strip()
        logging.info(f"Intro script ({target_lang}, {len(condensed_script.split())} words): {condensed_script!r}")
    except Exception as e:
        logging.error(f"LLM condensation failed: {e}")

    if not condensed_script:
        condensed_script = " ".join(intro_text.split()[:25])
        logging.info(f"Fallback script: {condensed_script!r}")

    # ── Step 2: Submit avatar video job ──────────────────────────────────────
    payload = {
        "type":         "avatar",
        "avatar_id":    avatar_id,
        "script":       condensed_script,
        "voice_id":     voice_id,
        "aspect_ratio": "16:9",
    }
    if bg_url:
        payload["background"] = {"type": "image", "url": bg_url}
        logging.info(f"Background image attached: {bg_url[:80]}")
    logging.info(f"Submitting HeyGen v3 avatar job (avatar={avatar_id}, voice={voice_id}, bg={'yes' if bg_url else 'none'})…")
    try:
        r = requests.post(
            "https://api.heygen.com/v3/videos",
            headers=HEYGEN_HEADERS,
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
    except Exception as e:
        logging.error(f"HeyGen submit failed: {e}")
        return ""

    video_id = r.json().get("data", {}).get("video_id", "")
    if not video_id:
        logging.error(f"HeyGen returned no video_id: {r.text[:300]}")
        return ""
    logging.info(f"HeyGen video_id: {video_id}")

    # ── Step 3: Poll until complete (max 3 minutes) ───────────────────────────
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(5)
        try:
            sr = requests.get(
                f"https://api.heygen.com/v3/videos/{video_id}",
                headers={"X-Api-Key": HEYGEN_API_KEY},
                timeout=15,
            )
            sr.raise_for_status()
        except Exception as e:
            logging.warning(f"HeyGen poll error (will retry): {e}")
            continue

        status_data = sr.json().get("data", {})
        status = status_data.get("status", "")
        logging.info(f"HeyGen status: {status}")

        if status == "completed":
            video_url = status_data.get("video_url", "")
            logging.info(f"HeyGen video ready: {video_url[:80]}")
            return video_url
        elif status == "failed":
            reason = (status_data.get("failure_message")
                      or status_data.get("failure_code")
                      or status_data.get("error")
                      or "unknown")
            logging.error(f"HeyGen job failed: {reason}")
            return ""

    logging.error("HeyGen polling timed out after 3 minutes")
    return ""

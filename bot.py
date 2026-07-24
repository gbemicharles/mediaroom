import asyncio
import os
import re
import logging
import subprocess
import uuid
import glob
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from config import check_env_vars, TELEGRAM_BOT_TOKEN, DEFAULT_PROMPT, get_webshare_proxies, WEBSHARE_PROXY_USERNAME, WEBSHARE_PROXY_PASSWORD
from ai_services import generate_text_and_extract_prompt, generate_thumbnail, generate_intro_video, translate_chunk, split_transcript_into_chunks

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# In-memory storage for user-specific prompts (resets on restart)
user_prompts = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = (
        "🤖 Welcome to the YouTube Automation Bot!\n\n"
        "Send me a `.txt` file containing your video's transcript. "
        "You can include the YouTube link in the caption of the file.\n\n"
        "I will generate:\n"
        "✅ 3 Title Ideas\n"
        "✅ Description\n"
        "✅ Hashtags & Tags\n"
        "✅ A Thumbnail Image (DALL-E 3)\n"
        "✅ A Short AI Intro Video (Replicate)\n\n"
        "Use `/setprompt <your prompt>` to override the default instructions."
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_msg)

async def set_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    new_prompt = " ".join(context.args)
    
    if not new_prompt:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Usage: `/setprompt your custom instructions here`")
        return
    
    # We still need the thumbnail tag instructions appended so our parsing doesn't break
    system_prompt = f"{new_prompt}\n\nCRITICAL INSTRUCTION: You must wrap the DALL-E 3 image generation prompt inside <thumbnail_prompt> tags. For example:\n<thumbnail_prompt>A highly engaging thumbnail...</thumbnail_prompt>"
    
    user_prompts[user_id] = system_prompt
    await context.bot.send_message(chat_id=update.effective_chat.id, text="✅ Custom prompt set successfully! It will be used for your next transcripts.")

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|\/|$)", url)
    if match:
        return match.group(1)
    match = re.search(r"youtu\.be\/([0-9A-Za-z_-]{11})", url)
    if match:
        return match.group(1)
    return None

def _make_session_with_proxy(proxy_url: str = None, timeout: int = 10):
    """Creates a requests.Session with optional proxy and a per-request timeout."""
    import requests as req
    from requests.adapters import HTTPAdapter

    class TimeoutAdapter(HTTPAdapter):
        def __init__(self, timeout, *args, **kwargs):
            self._timeout = timeout
            super().__init__(*args, **kwargs)
        def send(self, *args, **kwargs):
            kwargs.setdefault("timeout", self._timeout)
            return super().send(*args, **kwargs)

    session = req.Session()
    session.mount("http://", TimeoutAdapter(timeout))
    session.mount("https://", TimeoutAdapter(timeout))
    if proxy_url:
        session.proxies = {"http": proxy_url, "https": proxy_url}
    return session


def download_transcript_direct(video_url: str) -> str:
    """
    Downloads transcript by fetching the YouTube watch page directly with browser-style
    headers through the Webshare rotating residential proxy, then pulling the caption XML.
    Avoids youtube-transcript-api's rapid-fire internal requests that trigger CAPTCHA.
    """
    import requests as req
    import xml.etree.ElementTree as ET
    import html

    video_id = extract_video_id(video_url)
    if not video_id:
        raise Exception("Could not extract video ID from URL")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }

    # Build proxy — rotating residential if available, else none
    proxies = None
    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
        proxy_url = f"http://{WEBSHARE_PROXY_USERNAME}-rotate:{WEBSHARE_PROXY_PASSWORD}@p.webshare.io:80"
        proxies = {"http": proxy_url, "https": proxy_url}

    # Step 1: fetch the watch page
    r = req.get(
        f"https://www.youtube.com/watch?v={video_id}",
        headers=headers, proxies=proxies, timeout=20
    )
    r.raise_for_status()

    # Step 2: extract caption track URLs from the embedded JSON
    import json
    match = re.search(r'"captionTracks":(\[.*?\])', r.text)
    if not match:
        raise Exception("No caption tracks found — video may have captions disabled.")

    tracks = json.loads(match.group(1))
    if not tracks:
        raise Exception("No caption tracks available for this video.")

    # Prefer English; fall back to first available track
    track = next((t for t in tracks if t.get("languageCode", "").startswith("en")), tracks[0])
    base_url = track.get("baseUrl")
    if not base_url:
        raise Exception("Could not extract caption URL.")

    # Step 3: fetch the transcript XML
    rx = req.get(base_url, headers=headers, proxies=proxies, timeout=15)
    rx.raise_for_status()

    # Step 4: parse <text> elements
    root = ET.fromstring(rx.text)
    lines = []
    for el in root.iter("text"):
        text = html.unescape("".join(el.itertext())).strip()
        text = re.sub(r"\s+", " ", text)
        if text and text.lower() not in ("[music]", "[applause]", "[laughter]"):
            lines.append(text)

    if not lines:
        raise Exception("Transcript was empty after parsing.")

    return " ".join(lines)

def download_transcript_invidious(video_url: str) -> str:
    """Downloads transcript via public Invidious instances (bypasses YouTube IP blocks)."""
    import requests as req
    video_id = extract_video_id(video_url)
    if not video_id:
        raise Exception("Could not extract video ID from URL")

    # Public Invidious instances to try in order
    instances = [
        "https://inv.nadeko.net",  # verified working 2026-07-24
    ]

    last_error = None
    for instance in instances:
        try:
            # Get list of available caption tracks
            captions_url = f"{instance}/api/v1/captions/{video_id}"
            resp = req.get(captions_url, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            captions = data.get("captions", [])
            if not captions:
                continue

            # Prefer English, fall back to first available
            track = next((c for c in captions if "en" in c.get("language_code", "").lower()), captions[0])
            subtitle_url = track.get("url")
            if not subtitle_url:
                continue

            # Invidious returns relative URLs — prepend instance
            if subtitle_url.startswith("/"):
                subtitle_url = instance + subtitle_url

            sub_resp = req.get(subtitle_url, timeout=10)
            sub_resp.raise_for_status()

            # Parse XML subtitle format
            import xml.etree.ElementTree as ET
            root = ET.fromstring(sub_resp.text)
            lines = []
            for p in root.iter("p"):
                text = "".join(p.itertext()).strip()
                if text and text.lower() != "[music]":
                    lines.append(text)

            if lines:
                return " ".join(lines)

        except Exception as e:
            last_error = e
            continue

    raise Exception(f"All Invidious instances failed. Last error: {last_error}")


def download_transcript_ytdlp(video_url: str) -> str:
    """Downloads transcript using yt-dlp with Chrome impersonation + Webshare proxy."""
    unique_id = str(uuid.uuid4())
    temp_template = os.path.join(os.getcwd(), f"temp_{unique_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--impersonate", "Chrome-136",
        "--write-auto-subs",
        "--write-subs",
        "--skip-download",
        "--sub-langs", "en.*,ro,es,fr,de,pt,it,zh,ja,ru,ar",  # grab English + common langs
        "-o", temp_template,
    ]

    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
        # Plain format (no -rotate suffix) keeps the same exit IP for the whole
        # yt-dlp process — required so the subtitle CDN URL (which is IP-bound)
        # is fetched from the same IP that requested it.
        stable_proxy = f"http://{WEBSHARE_PROXY_USERNAME}:{WEBSHARE_PROXY_PASSWORD}@p.webshare.io:80"
        cmd += ["--proxy", stable_proxy]
    else:
        proxy_urls = get_webshare_proxies()
        if proxy_urls:
            cmd += ["--proxy", proxy_urls[0]]

    cmd.append(video_url)
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Don't fail on non-zero exit — yt-dlp exits non-zero if even one subtitle
    # variant gets a 429, even when other tracks downloaded successfully.
        
    # Find downloaded subtitle files (yt-dlp may exit non-zero but still write some files)
    pattern = os.path.join(os.getcwd(), f"temp_{unique_id}.*")
    downloaded_files = glob.glob(pattern)
    if not downloaded_files:
        error_msg = res.stderr or res.stdout or "No subtitle files downloaded"
        raise Exception(f"yt-dlp failed to download subtitles: {error_msg}")
        
    vtt_file = downloaded_files[0]
    
    try:
        with open(vtt_file, "r", encoding="utf-8") as f:
            vtt_content = f.read()
    finally:
        # Clean up all downloaded temp files matching this ID
        for f in downloaded_files:
            try:
                os.remove(f)
            except Exception:
                pass
                
    # Parse VTT content
    lines = vtt_content.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:') or '-->' in line:
            continue
        line = re.sub(r'<[^>]+>', '', line)
        line = line.strip()
        if line and line.lower() != '[music]':
            if cleaned_lines and (line in cleaned_lines[-1] or cleaned_lines[-1] in line):
                if len(line) > len(cleaned_lines[-1]):
                    cleaned_lines[-1] = line
            else:
                cleaned_lines.append(line)
                
    return " ".join(cleaned_lines)

def get_language_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("🇺🇸 English", callback_data="lang_English"),
            InlineKeyboardButton("🇪🇸 Español (Spanish)", callback_data="lang_Spanish")
        ],
        [
            InlineKeyboardButton("🇫🇷 Français (French)", callback_data="lang_French"),
            InlineKeyboardButton("🇩🇪 Deutsch (German)", callback_data="lang_German")
        ],
        [
            InlineKeyboardButton("🇧🇷 Português (Portuguese)", callback_data="lang_Portuguese"),
            InlineKeyboardButton("🇮🇹 Italiano (Italian)", callback_data="lang_Italian")
        ],
        [
            InlineKeyboardButton("🇨🇳 中文 (Chinese)", callback_data="lang_Chinese"),
            InlineKeyboardButton("🇯🇵 日本語 (Japanese)", callback_data="lang_Japanese")
        ],
        [
            InlineKeyboardButton("🇷🇺 Русский (Russian)", callback_data="lang_Russian"),
            InlineKeyboardButton("🇵🇱 Polski (Polish)", callback_data="lang_Polish")
        ],
        [
            InlineKeyboardButton("🇷🇴 Română (Romanian)", callback_data="lang_Romanian"),
            InlineKeyboardButton("🇹🇷 Türkçe (Turkish)", callback_data="lang_Turkish")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def _parse_production_pack(full_text: str) -> list[tuple[str, str]]:
    """Parse the AI production pack into (header, body) section pairs for pretty formatting."""
    SECTION_EMOJIS = {
        "TITLE": "🏆", "WINNER": "🥇", "SEO": "📋", "DESCRIPTION": "📋",
        "HASHTAG": "#️⃣", "TAG": "🏷️", "HOST SCRIPT": "🎙️", "SCRIPT": "🎙️",
        "PHOTO PROMPT": "📸", "THUMBNAIL": "🖼️",
    }

    def emoji_for(title: str) -> str:
        upper = title.upper()
        for key, em in SECTION_EMOJIS.items():
            if key in upper:
                return em
        return "▪️"

    # Split on lines that look like  "# 1. TITLE IDEAS" or "## 1. TITLE IDEAS"
    parts = re.split(r'\n(?=#{1,3}\s*\d+[\.\)]\s+)', full_text.strip())
    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        header_match = re.match(r'^#{1,3}\s*\d+[\.\)]\s+(.+)', part)
        if header_match:
            title = header_match.group(1).strip()
            body = part[header_match.end():].strip()
            em = emoji_for(title)
            sections.append((f"{em} {title}", body))
        else:
            # Preamble / unlabeled text — skip if very short
            if len(part) > 30:
                sections.append(("", part))
    return sections


def _html(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def _send_html(bot, chat_id: int, text: str):
    """Send a message with HTML parse mode, splitting if over 4000 chars."""
    MAX = 4000
    if len(text) <= MAX:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
        return
    # Split on double newlines to avoid breaking mid-word
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > MAX:
            if current:
                await bot.send_message(chat_id=chat_id, text=current.strip(), parse_mode="HTML")
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        await bot.send_message(chat_id=chat_id, text=current.strip(), parse_mode="HTML")


async def process_transcript(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, transcript_text: str, target_lang: str):
    try:
        prompt_to_use = user_prompts.get(user_id, DEFAULT_PROMPT)

        # Split transcript into 10,000-char chunks at sentence boundaries
        CHUNK_SIZE = 10000
        chunks = split_transcript_into_chunks(transcript_text, max_chars=CHUNK_SIZE)
        total_chunks = len(chunks)

        if total_chunks > 1:
            await _send_html(
                context.bot, chat_id,
                f"🧠 <b>Generating production pack in {_html(target_lang)}…</b>\n"
                f"<i>Transcript split into {total_chunks} parts — translating all of them. This may take a few minutes.</i>"
            )
        else:
            await _send_html(
                context.bot, chat_id,
                f"🧠 <b>Generating production pack in {_html(target_lang)}…</b>\n"
                f"<i>Translating transcript, writing scripts, building assets — hang tight.</i>"
            )
        logging.info(f"STEP 1: {total_chunks} chunk(s). Total chars: {len(transcript_text)}")

        # Chunk 1 → full production pack + translated first part
        full_text, image_prompt, translated_part1 = await asyncio.to_thread(
            generate_text_and_extract_prompt, chunks[0], prompt_to_use, target_lang
        )
        logging.info(f"STEP 2: Chunk 1 done. pack={len(full_text)} chars, transcript_part={len(translated_part1)} chars")

        # Remaining chunks → translation only
        translated_parts = [translated_part1] if translated_part1 else []
        for idx, chunk in enumerate(chunks[1:], start=2):
            await _send_html(
                context.bot, chat_id,
                f"🔄 <b>Translating part {idx}/{total_chunks}…</b>"
            )
            logging.info(f"STEP 2.{idx}: Translating chunk {idx}/{total_chunks} ({len(chunk)} chars)")
            part = await asyncio.to_thread(translate_chunk, chunk, target_lang)
            translated_parts.append(part)
            logging.info(f"STEP 2.{idx}: Done ({len(part)} chars)")

        translated_transcript = "\n\n".join(translated_parts)
        logging.info(f"STEP 3: Combined transcript: {len(translated_transcript)} chars across {len(translated_parts)} parts")

        # ── Translated transcript .txt ───────────────────────────────────────
        if translated_transcript.strip():
            filename = f"transcript_{target_lang.replace(' ', '_')}.txt"
            filepath = os.path.join(os.getcwd(), filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(translated_transcript)
            try:
                with open(filepath, "rb") as f:
                    await context.bot.send_document(
                        chat_id=chat_id,
                        document=f,
                        filename=filename,
                        caption=f"📄 <b>Translated &amp; Rewritten Transcript</b> — {_html(target_lang)}",
                        parse_mode="HTML"
                    )
            except Exception as e:
                logging.error(f"Error sending transcript document: {e}")
                await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Could not send transcript file: {e}")
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            logging.warning("STEP 3: No translated transcript — chunk 1 may have run out of tokens.")

        # ── Production pack ──────────────────────────────────────────────────
        logging.info(f"STEP 4: Sending production pack ({len(full_text)} chars)")
        if not full_text.strip():
            await context.bot.send_message(chat_id=chat_id, text="⚠️ AI returned an empty response. Please try again.")
            return

        # Header card
        await _send_html(
            context.bot, chat_id,
            f"🎬 <b>PRODUCTION PACK</b> — <b>{_html(target_lang)}</b>\n"
            f"{'─' * 28}"
        )

        sections = _parse_production_pack(full_text)
        if sections:
            for header, body in sections:
                if not body.strip():
                    continue
                msg = (f"<b>{_html(header)}</b>\n{'─' * 20}\n{_html(body)}" if header
                       else _html(body))
                await _send_html(context.bot, chat_id, msg)
        else:
            # Fallback: send raw text if parsing found nothing
            await _send_html(context.bot, chat_id, _html(full_text))

        # ── Thumbnail ────────────────────────────────────────────────────────
        logging.info(f"STEP 5: image_prompt present: {bool(image_prompt)}")
        if image_prompt and image_prompt != "A generic YouTube thumbnail":
            await _send_html(
                context.bot, chat_id,
                f"🎨 <b>Generating thumbnail…</b>\n<i>{_html(image_prompt[:200])}</i>"
            )
            logging.info("STEP 6: Calling FLUX Schnell")
            image_url = await asyncio.to_thread(generate_thumbnail, image_prompt)
            logging.info(f"STEP 7: Thumbnail URL: {image_url[:80] if image_url else 'EMPTY'}")

            if image_url:
                await context.bot.send_photo(
                    chat_id=chat_id, photo=image_url,
                    caption="🖼️ <b>Generated Thumbnail</b>", parse_mode="HTML"
                )

                await _send_html(context.bot, chat_id, "🎬 <b>Generating intro video…</b> <i>(~30–60 s)</i>")
                logging.info("STEP 8: Calling SVD video generation")
                video_url = await asyncio.to_thread(generate_intro_video, image_url)
                logging.info(f"STEP 9: Video URL: {video_url[:80] if video_url else 'EMPTY'}")

                if video_url:
                    await context.bot.send_video(
                        chat_id=chat_id, video=video_url,
                        caption="🎞️ <b>Generated Intro Video</b>", parse_mode="HTML"
                    )
                else:
                    await _send_html(context.bot, chat_id, "❌ <b>Intro video generation failed.</b>")
            else:
                await _send_html(context.bot, chat_id, "❌ <b>Thumbnail generation failed.</b> Check Replicate billing.")
        else:
            await _send_html(context.bot, chat_id, "⚠️ <b>No thumbnail prompt found</b> in the AI response.")

        logging.info("STEP 10: process_transcript complete")

    except Exception as e:
        logging.error(f"Error in process_transcript: {e}", exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text=f"❌ An error occurred: {e}")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    document = update.message.document
    caption = update.message.caption or ""

    if not document.file_name.endswith('.txt'):
        await context.bot.send_message(chat_id=chat_id, text="Please send a `.txt` file containing the transcript.")
        return

    await context.bot.send_message(chat_id=chat_id, text="📥 Received transcript. Downloading...")

    try:
        # Download file
        file = await context.bot.get_file(document.file_id)
        # Using a temporary file path
        temp_path = f"transcript_{user_id}.txt"
        await file.download_to_drive(temp_path)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
            
        os.remove(temp_path)
        
        # Cache transcript in user_data
        context.user_data['pending_transcript'] = transcript_text
        context.user_data['pending_video_url'] = None
        context.user_data['pending_caption'] = caption
        
        await context.bot.send_message(
            chat_id=chat_id,
            text="🌐 Select the target language for the transcript translation and generated assets:",
            reply_markup=get_language_keyboard()
        )
    except Exception as e:
        logging.error(f"Error handling document: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"❌ An error occurred: {e}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    text = update.message.text or ""
    
    video_id = extract_video_id(text)
    if video_id:
        # Cache video URL in user_data
        context.user_data['pending_video_url'] = text
        context.user_data['pending_transcript'] = None
        context.user_data['pending_caption'] = ""
    else:
        # Cache text directly as transcript/input content
        context.user_data['pending_video_url'] = None
        context.user_data['pending_transcript'] = text
        context.user_data['pending_caption'] = ""
    
    await context.bot.send_message(
        chat_id=chat_id,
        text="🌐 Select the target language for the transcript translation and generated assets:",
        reply_markup=get_language_keyboard()
    )

async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    target_lang = query.data.split('_')[1]
    
    video_url = context.user_data.get('pending_video_url')
    transcript_text = context.user_data.get('pending_transcript')
    caption = context.user_data.get('pending_caption', '')
    
    if not video_url and not transcript_text:
        await query.edit_message_text("❌ No active request found. Please send a YouTube link or a `.txt` file.")
        return
        
    await query.edit_message_text(f"⏳ Language chosen: {target_lang}. Starting process...")
    
    # Clear session cache
    context.user_data['pending_video_url'] = None
    context.user_data['pending_transcript'] = None
    context.user_data['pending_caption'] = None
    
    if video_url:
        await context.bot.send_message(chat_id=chat_id, text="📥 Detected YouTube link. Downloading original transcript automatically...")

        def _fetch_transcript():
            """Try all download methods sequentially (runs in a thread)."""
            try:
                return download_transcript_ytdlp(video_url)
            except Exception as e1:
                logging.info(f"yt-dlp failed: {e1}. Trying direct fetch...")
            try:
                return download_transcript_direct(video_url)
            except Exception as e2:
                logging.info(f"Direct fetch failed: {e2}. Trying Invidious...")
            try:
                return download_transcript_invidious(video_url)
            except Exception as e3:
                logging.error(f"All transcript downloaders failed: {e3}")
            return None

        try:
            transcript_text = await asyncio.wait_for(
                asyncio.to_thread(_fetch_transcript),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            transcript_text = None

        if not transcript_text:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "❌ Could not download transcript automatically.\n\n"
                    "YouTube is rate-limiting requests right now (temporary — usually resets within 1-2 hours).\n\n"
                    "👉 In the meantime:\n"
                    "1. Copy & paste the transcript/script as a text message.\n"
                    "2. Or upload a .txt file with the transcript.\n\n"
                    "In normal use (a few videos per day) this error won't appear."
                )
            )
            return
            
    # Combine caption context if transcript came from file
    if caption:
        transcript_text = f"Additional User Info/Link: {caption}\n\nTranscript:\n{transcript_text}"
        
    # Process
    await process_transcript(context, chat_id, user_id, transcript_text, target_lang)

if __name__ == '__main__':
    # Ensure environment variables are loaded
    check_env_vars()
    
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('setprompt', set_prompt))
    application.add_handler(CallbackQueryHandler(language_callback))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("Bot is starting...")
    application.run_polling()


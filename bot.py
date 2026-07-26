import asyncio
import os
import re
import logging
import subprocess
import uuid
import glob


# ── Custom transcript-fetch exceptions ──────────────────────────────────────
class NoSubtitlesError(Exception):
    """Raised when a video genuinely has no subtitles/captions available."""

class AgeRestrictedError(Exception):
    """Raised when a video is age-restricted and cannot be accessed automatically."""

class PrivateVideoError(Exception):
    """Raised when a video is private or unavailable."""

class RateLimitedError(Exception):
    """Raised when YouTube is rate-limiting requests (HTTP 429)."""
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from config import check_env_vars, TELEGRAM_BOT_TOKEN, DEFAULT_PROMPT, get_webshare_proxies, WEBSHARE_PROXY_USERNAME, WEBSHARE_PROXY_PASSWORD
from ai_services import generate_text_and_extract_prompt, generate_thumbnail, generate_intro_video, translate_chunk, split_transcript_into_chunks, split_into_paragraphs

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# In-memory storage for user-specific prompts (resets on restart)
user_prompts = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = (
        "🎬 Welcome to <b>Mediaroom</b>!\n\n"
        "Send me a <b>YouTube link</b> or a <b>.txt transcript file</b> and I'll produce a full content pack:\n\n"
        "✅ 3 Title Ideas\n"
        "✅ SEO Description\n"
        "✅ Hashtags &amp; Tags\n"
        "✅ AI Host Script\n"
        "✅ Translated Transcript\n"
        "✅ Thumbnail Image\n"
        "✅ Short Intro Video\n\n"
        "Use <code>/setprompt &lt;your prompt&gt;</code> to override the default instructions."
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_msg, parse_mode="HTML")

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

    # Build proxy — stable (non-rotating) so the caption URL is fetched from the
    # same exit IP that fetched the watch page (YouTube binds caption URLs to IP).
    proxies = None
    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
        proxy_url = f"http://{WEBSHARE_PROXY_USERNAME}:{WEBSHARE_PROXY_PASSWORD}@p.webshare.io:80"
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
        raise NoSubtitlesError("No caption tracks found — video may have captions disabled.")

    tracks = json.loads(match.group(1))
    if not tracks:
        raise NoSubtitlesError("No caption tracks available for this video.")

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
    no_captions_count = 0
    for instance in instances:
        try:
            # Get list of available caption tracks
            captions_url = f"{instance}/api/v1/captions/{video_id}"
            resp = req.get(captions_url, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            # Invidious returns {"error": "<message>"} for hard failures.
            # Detect private / age-restricted / unavailable immediately so we
            # don't waste time on slower fallback methods.
            api_error = data.get("error", "")
            if api_error:
                err_lower = api_error.lower()
                if any(p in err_lower for p in ("private", "unavailable", "has been removed",
                                                  "account associated")):
                    raise PrivateVideoError(f"This video is private or unavailable: {api_error}")
                if any(p in err_lower for p in ("age-restricted", "age restricted",
                                                  "sign in to confirm your age",
                                                  "confirm your age", "inappropriate for some users")):
                    raise AgeRestrictedError(f"This video is age-restricted: {api_error}")
                # Other hard API errors — skip to next instance
                last_error = Exception(f"Invidious API error: {api_error}")
                continue

            captions = data.get("captions", [])
            if not captions:
                no_captions_count += 1
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

        except (PrivateVideoError, AgeRestrictedError, NoSubtitlesError):
            raise  # hard errors — do not try other instances
        except Exception as e:
            last_error = e
            continue

    # If every reachable instance reported an empty captions list, the video has no captions
    if no_captions_count > 0 and no_captions_count == len(instances):
        raise NoSubtitlesError("This video has no captions available.")

    raise Exception(f"All Invidious instances failed. Last error: {last_error}")


def download_transcript_ytapi(video_url: str) -> str:
    """Downloads transcript using youtube-transcript-api v1.2.4 with Webshare proxy."""
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        NoTranscriptFound,
        TranscriptsDisabled,
        AgeRestricted,
        VideoUnavailable,
        VideoUnplayable,
        IpBlocked,
        RequestBlocked,
    )

    video_id = extract_video_id(video_url)
    if not video_id:
        raise Exception("Could not extract video ID from URL")

    proxy_config = None
    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
        from youtube_transcript_api.proxies import WebshareProxyConfig
        # Rotating residential pool — required because YouTube blocks static IPs.
        # retries_when_blocked=10 means it auto-rotates to a fresh IP on each block.
        proxy_config = WebshareProxyConfig(
            proxy_username=WEBSHARE_PROXY_USERNAME,
            proxy_password=WEBSHARE_PROXY_PASSWORD,
            retries_when_blocked=10,
        )

    api = YouTubeTranscriptApi(proxy_config=proxy_config)

    PREF_LANGS = ['en', 'ro', 'de', 'fr', 'es', 'pt', 'it', 'ru', 'ar', 'uk', 'nl', 'pl', 'tr', 'zh', 'ja']

    try:
        fetched = None
        last_err = None
        for attempt in range(3):
            try:
                tl = api.list(video_id)
                try:
                    fetched = tl.find_transcript(PREF_LANGS).fetch()
                except Exception:
                    fetched = next(iter(tl)).fetch()
                break  # success
            except Exception as e:
                last_err = e
                err_str = str(e)
                # Retry only on transient connection errors; bail immediately on hard errors
                if any(x in err_str for x in ("IncompleteRead", "Connection broken", "RemoteDisconnected", "ConnectionError")):
                    logging.warning(f"ytapi attempt {attempt+1} transient error: {e}. Retrying...")
                    import time; time.sleep(1)
                    continue
                # Short-circuit on definitive hard errors — retrying or calling api.fetch() is pointless
                if isinstance(e, (NoTranscriptFound, TranscriptsDisabled, AgeRestricted,
                                  VideoUnavailable, VideoUnplayable, IpBlocked, RequestBlocked)):
                    raise
                # For other non-transient errors try fetch() directly as last resort
                try:
                    fetched = api.fetch(video_id, languages=PREF_LANGS)
                    break
                except Exception:
                    raise  # propagate so outer try/except catches it

        if fetched is None and last_err is not None:
            raise last_err

    except (NoTranscriptFound, TranscriptsDisabled) as e:
        raise NoSubtitlesError("This video has no captions available.") from e
    except AgeRestricted as e:
        raise AgeRestrictedError("This video is age-restricted.") from e
    except (VideoUnavailable, VideoUnplayable) as e:
        raise PrivateVideoError("This video is private or unavailable.") from e
    except (IpBlocked, RequestBlocked) as e:
        raise RateLimitedError("YouTube is rate-limiting requests.") from e

    lines = []
    for entry in fetched:
        text = entry.get('text', '') if isinstance(entry, dict) else getattr(entry, 'text', str(entry))
        text = re.sub(r'\s+', ' ', text.strip())
        if text and text.lower() not in ('[music]', '[applause]', '[laughter]'):
            lines.append(text)

    if not lines:
        raise Exception("Transcript was empty after parsing.")

    return ' '.join(lines)


def _run_ytdlp(video_url: str, unique_id: str, proxy: str = None) -> tuple:
    """Run yt-dlp and return (list of downloaded subtitle file paths, stderr text)."""
    temp_template = os.path.join(os.getcwd(), f"temp_{unique_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--impersonate", "Chrome-136",
        "--write-auto-subs",
        "--write-subs",
        "--skip-download",
        "--ignore-errors",   # continue past 429s on individual subtitle variants
        "--sub-langs", "en.*,ro,de,fr,es,pt,it,ru,ar,uk,nl,pl,tr,zh,ja,he,hi,hu,cs,sv,no,da,fi",
        "-o", temp_template,
    ]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd.append(video_url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = (result.stderr or "") + (result.stdout or "")
    pattern = os.path.join(os.getcwd(), f"temp_{unique_id}.*")
    return glob.glob(pattern), stderr


def _classify_ytdlp_stderr(stderr: str):
    """Inspect yt-dlp output and raise a specific exception if a hard failure is detected.
    Returns None when the output is ambiguous (allow caller to fall through)."""
    s = stderr.lower()
    # Private / unavailable
    if any(p in s for p in ("private video", "this video is private", "video is unavailable",
                             "has been removed", "account associated with this video")):
        raise PrivateVideoError("This video is private or unavailable.")
    # Age-restricted
    if any(p in s for p in ("sign in to confirm your age", "age-restricted",
                             "age_verification", "inappropriate for some users")):
        raise AgeRestrictedError("This video is age-restricted.")
    # No subtitles
    if any(p in s for p in ("no subtitles", "there are no subtitles",
                             "no automatic captions", "subtitles not available",
                             "this video does not have", "couldn't find automatic captions")):
        raise NoSubtitlesError("This video has no captions.")
    # Rate-limited
    if "http error 429" in s or "too many requests" in s:
        raise RateLimitedError("YouTube is rate-limiting requests.")


def download_transcript_ytdlp(video_url: str) -> str:
    """Downloads transcript using yt-dlp with Chrome impersonation.
    Tries with Webshare stable proxy first; retries without proxy if no files downloaded."""
    unique_id = str(uuid.uuid4())

    stable_proxy = None
    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
        stable_proxy = f"http://{WEBSHARE_PROXY_USERNAME}:{WEBSHARE_PROXY_PASSWORD}@p.webshare.io:80"

    downloaded_files, stderr1 = _run_ytdlp(video_url, unique_id, proxy=stable_proxy)

    # If proxy got nothing, classify stderr first; only retry without proxy on ambiguous errors
    if not downloaded_files and stable_proxy:
        _classify_ytdlp_stderr(stderr1)   # raises hard errors (private, age-restricted, no-subs, 429)
        logging.info("yt-dlp with proxy got no files — retrying without proxy")
        downloaded_files, stderr2 = _run_ytdlp(video_url, unique_id + "_noproxy", proxy=None)
        combined_stderr = stderr1 + "\n" + stderr2
    else:
        combined_stderr = stderr1

    if not downloaded_files:
        # One last classification pass before raising a generic error
        _classify_ytdlp_stderr(combined_stderr)
        raise Exception("yt-dlp failed to download subtitles: no subtitle files found")

    # Prefer English subtitle files; fall back to whatever was downloaded first
    def _lang_priority(path):
        name = os.path.basename(path).lower()
        if ".en." in name or name.endswith(".en.vtt"):
            return 0
        if ".en-" in name:
            return 1
        return 2

    vtt_file = sorted(downloaded_files, key=_lang_priority)[0]
    
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
        "TITLE": "🏆", "BEST TITLE": "🥇", "WINNER": "🥇", "SEO": "📋", "DESCRIPTION": "📋",
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
            # Normalise legacy label
            if title.upper() == "WINNER":
                title = "BEST TITLE"
            body = part[header_match.end():].strip()
            em = emoji_for(title)
            sections.append((f"{em} {title}", body))
        else:
            # Preamble / unlabeled text — skip if very short
            if len(part) > 30:
                sections.append(("", part))
    return sections


def _html(text: str) -> str:
    """Escape text for Telegram HTML parse mode, converting common markdown to HTML tags."""
    # Escape HTML special chars first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Convert markdown bold/italic to HTML (after escaping so we don't double-escape)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text, flags=re.DOTALL)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text, flags=re.DOTALL)
    # Strip leftover lone asterisks (e.g. bullet points written as "* item")
    text = re.sub(r'(?m)^\* ', '• ', text)
    return text


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

        await _send_html(
            context.bot, chat_id,
            f"🧠 <b>Generating production pack in {_html(target_lang)}…</b>\n"
            + (f"<i>Transcript split into {total_chunks} parts — all translating in parallel.</i>"
               if total_chunks > 1 else
               "<i>Translating transcript, writing scripts, building assets — hang tight.</i>")
        )
        logging.info(f"STEP 1: {total_chunks} chunk(s), {len(transcript_text)} total chars. Launching parallel tasks.")

        async def _translate_with_retry(chunk: str, lang: str, max_retries: int = 2) -> str:
            """Run translate_chunk in a thread; retry up to max_retries times on failure."""
            for attempt in range(1, max_retries + 1):
                try:
                    result = await asyncio.to_thread(translate_chunk, chunk, lang)
                    if result and result.strip():
                        return result
                    logging.warning(f"translate_chunk attempt {attempt} returned empty — retrying")
                except Exception as e:
                    logging.warning(f"translate_chunk attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2)
            logging.error("All translate_chunk retries exhausted — using original text")
            return chunk

        # Fire everything simultaneously:
        #   • production pack from chunk 0 (titles, SEO, scripts, etc.)
        #   • translate_chunk (Haiku) for every chunk in parallel, with retry
        pack_task = asyncio.to_thread(
            generate_text_and_extract_prompt, chunks[0], prompt_to_use, target_lang
        )
        translate_tasks = [
            _translate_with_retry(chunk, target_lang)
            for chunk in chunks
        ]

        results = await asyncio.gather(pack_task, *translate_tasks)

        pack_result = results[0]
        translated_parts = list(results[1:])   # one per chunk, in order

        full_text, image_prompt, pack_transcript = pack_result
        logging.info(f"STEP 2: pack={len(full_text)} chars, {len(translated_parts)} translated chunks")

        # Combine raw Haiku translations, then split into ≤999-char paragraphs
        translated_transcript_raw = "\n\n".join(p for p in translated_parts if p.strip())
        paragraphs = split_into_paragraphs(translated_transcript_raw, min_chars=500, max_chars=999)

        # Prepend the channel intro translated by Sonnet (first paragraph of pack_transcript)
        if pack_transcript:
            intro = pack_transcript.split("\n\n")[0].strip()
            if intro:
                paragraphs.insert(0, intro)

        translated_transcript = "\n\n".join(paragraphs)
        logging.info(f"STEP 3: Combined transcript: {len(translated_transcript)} chars, {len(paragraphs)} paragraphs")

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
                logging.info("STEP 8: Calling Wan 2.1 i2v video generation")
                video_url = await asyncio.to_thread(generate_intro_video, image_url, image_prompt)
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

        # ── Summary card ─────────────────────────────────────────────────────
        has_thumbnail = bool(image_prompt and image_prompt != "A generic YouTube thumbnail")
        summary_lines = [
            f"✅ <b>Production pack complete!</b>",
            f"",
            f"📄 Transcript — <b>{total_chunks}</b> part(s) translated &amp; combined",
            f"🎬 Production pack — titles, SEO, hashtags, host script",
            f"🖼️ Thumbnail — {'generated' if has_thumbnail else '⚠️ skipped (no prompt)'}",
            f"🎞️ Intro video — {'generated' if has_thumbnail else '⚠️ skipped'}",
            f"",
            f"<i>Send another transcript or YouTube link to start a new job.</i>",
        ]
        await _send_html(context.bot, chat_id, "\n".join(summary_lines))
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

def _is_meaningful_transcript(text: str) -> bool:
    """Return True if text looks like a real transcript (≥200 chars and ≥40 words)."""
    stripped = text.strip()
    if len(stripped) < 200:
        return False
    word_count = len(stripped.split())
    return word_count >= 40


def _is_youtube_url(url: str) -> bool:
    return bool(re.search(r'(youtube\.com|youtu\.be)', url, re.IGNORECASE))


def _check_video_url(url: str) -> dict:
    """Probe url with yt-dlp --list-subs (no download) to decide if it is a
    supported video that has captions available.

    Returns a dict:
      supported   bool  – yt-dlp recognises the site / video
      has_captions bool – at least one caption/subtitle track exists
      error       str  – 'private' | 'age_restricted' | 'no_captions' |
                         'unsupported' | 'timeout' | '' (success)
    """
    proxy = None
    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
        proxy = f"http://{WEBSHARE_PROXY_USERNAME}:{WEBSHARE_PROXY_PASSWORD}@p.webshare.io:80"

    cmd = [
        "yt-dlp",
        "--list-subs",
        "--skip-download",
        "--no-warnings",
        "--impersonate", "Chrome-136",
    ]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd.append(url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        return {'supported': False, 'has_captions': False, 'error': 'timeout'}

    output = (result.stdout or "") + "\n" + (result.stderr or "")
    low = output.lower()

    # Hard errors — video exists on the platform but can't be used
    if any(p in low for p in ("private video", "this video is private",
                               "video is unavailable", "has been removed",
                               "account associated with this video")):
        return {'supported': True, 'has_captions': False, 'error': 'private'}

    if any(p in low for p in ("sign in to confirm your age", "age-restricted",
                               "age_verification", "inappropriate for some users")):
        return {'supported': True, 'has_captions': False, 'error': 'age_restricted'}

    # Unsupported / unrecognised URL
    if "unsupported url" in low or "is not a valid url" in low:
        return {'supported': False, 'has_captions': False, 'error': 'unsupported'}

    # Generic error with no caption info at all → treat as unsupported
    if result.returncode != 0 and "error" in low and "subtitle" not in low and "caption" not in low:
        return {'supported': False, 'has_captions': False, 'error': 'unsupported'}

    # Explicit "no captions"
    if any(p in low for p in ("has no subtitles", "no subtitles", "no automatic captions",
                               "subtitles not available", "does not have captions")):
        return {'supported': True, 'has_captions': False, 'error': 'no_captions'}

    # Captions found
    if any(p in low for p in ("available subtitles", "available automatic captions",
                               "language  formats", "language formats")):
        return {'supported': True, 'has_captions': True, 'error': ''}

    # Ambiguous output but no error — let the main fetch attempt it
    return {'supported': True, 'has_captions': True, 'error': ''}


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text or ""

    # 1. Does the message contain any URL?
    url_match = re.search(r'https?://\S+', text)
    if url_match:
        url = url_match.group(0).rstrip('.,;:!?)')  # strip trailing punctuation

        checking_msg = await context.bot.send_message(
            chat_id=chat_id,
            text="🔍 Checking link, please wait…"
        )

        check = await asyncio.to_thread(_check_video_url, url)

        if not check['supported']:
            err = check['error']
            if err == 'timeout':
                reply = (
                    "⏱ <b>The link took too long to check.</b>\n\n"
                    "Please try again, or send the transcript as a <code>.txt</code> file."
                )
            else:
                reply = (
                    "❌ <b>That doesn't appear to be a supported video link.</b>\n\n"
                    "Mediaroom can fetch transcripts from YouTube, Vimeo, Dailymotion, "
                    "and hundreds of other video platforms. If the video is valid, try "
                    "uploading its transcript as a <code>.txt</code> file instead."
                )
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=checking_msg.message_id,
                text=reply, parse_mode="HTML"
            )
            return

        if not check['has_captions']:
            err = check['error']
            if err == 'private':
                reply = (
                    "🔒 <b>This video is private or unavailable.</b>\n\n"
                    "Only public videos with captions can be processed automatically.\n\n"
                    "👉 Upload a <code>.txt</code> file with the transcript, "
                    "or paste the script as a text message."
                )
            elif err == 'age_restricted':
                reply = (
                    "🔞 <b>This video is age-restricted.</b>\n\n"
                    "It can't be accessed without a logged-in account.\n\n"
                    "👉 Upload a <code>.txt</code> file with the transcript, "
                    "or paste the script as a text message."
                )
            else:  # no_captions
                reply = (
                    "📭 <b>This video has no captions or subtitles.</b>\n\n"
                    "Automatic transcript extraction isn't possible without captions.\n\n"
                    "👉 Upload a <code>.txt</code> file with the transcript, "
                    "or paste the script as a text message."
                )
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=checking_msg.message_id,
                text=reply, parse_mode="HTML"
            )
            return

        # Valid video with captions — proceed
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=checking_msg.message_id,
            text="✅ Video found with captions. Select the target language:"
        )
        context.user_data['pending_video_url'] = url
        context.user_data['pending_transcript'] = None
        context.user_data['pending_caption'] = ""
        await context.bot.send_message(
            chat_id=chat_id,
            text="🌐 Select the target language for the transcript translation and generated assets:",
            reply_markup=get_language_keyboard()
        )
        return

    # 2. Meaningful transcript pasted as plain text?
    if _is_meaningful_transcript(text):
        context.user_data['pending_video_url'] = None
        context.user_data['pending_transcript'] = text
        context.user_data['pending_caption'] = ""
        await context.bot.send_message(
            chat_id=chat_id,
            text="🌐 Select the target language for the transcript translation and generated assets:",
            reply_markup=get_language_keyboard()
        )
        return

    # 3. Too short / meaningless
    word_count = len(text.strip().split())
    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            f"❌ That message is too short to process ({word_count} word{'s' if word_count != 1 else ''}).\n\n"
            "<b>Mediaroom</b> accepts:\n"
            "• A <b>video link</b> (YouTube, Vimeo, Dailymotion, and many more)\n"
            "• A <b>.txt file</b> — attach your transcript as a file\n"
            "• A <b>full transcript</b> pasted as text — at least 40 words"
        ),
        parse_mode="HTML"
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
        is_yt = _is_youtube_url(video_url)
        await context.bot.send_message(
            chat_id=chat_id,
            text="📥 Fetching transcript — this can take up to 60 seconds, please wait…"
        )

        def _fetch_transcript():
            """Try download methods sequentially (runs in a thread).
            Hard errors (no subtitles, age-restricted, private) are propagated
            immediately. YouTube-specific fallbacks are skipped for non-YouTube URLs."""
            rate_error = None

            try:
                return download_transcript_ytdlp(video_url)
            except (NoSubtitlesError, AgeRestrictedError, PrivateVideoError) as e:
                logging.info(f"yt-dlp hard error (not retrying): {e}")
                raise
            except RateLimitedError as e:
                rate_error = e
                logging.info(f"yt-dlp rate-limited: {e}.")
                if not is_yt:
                    raise rate_error
                logging.info("Trying direct fetch...")
            except Exception as e1:
                logging.info(f"yt-dlp failed: {e1}.")
                if not is_yt:
                    raise
                logging.info("Trying direct fetch...")

            # YouTube-only fallbacks
            try:
                return download_transcript_direct(video_url)
            except (NoSubtitlesError, AgeRestrictedError, PrivateVideoError) as e:
                logging.info(f"Direct fetch hard error: {e}")
                raise
            except RateLimitedError as e:
                rate_error = rate_error or e
                logging.info(f"Direct fetch rate-limited: {e}. Trying Invidious...")
            except Exception as e2:
                logging.info(f"Direct fetch failed: {e2}. Trying Invidious...")

            try:
                return download_transcript_invidious(video_url)
            except (NoSubtitlesError, AgeRestrictedError, PrivateVideoError) as e:
                logging.info(f"Invidious hard error: {e}")
                raise
            except RateLimitedError as e:
                rate_error = rate_error or e
                logging.info(f"Invidious rate-limited: {e}. Trying ytapi...")
            except Exception as e3:
                logging.info(f"Invidious failed: {e3}. Trying ytapi...")

            try:
                return download_transcript_ytapi(video_url)
            except (NoSubtitlesError, AgeRestrictedError, PrivateVideoError) as e:
                logging.info(f"ytapi hard error: {e}")
                raise
            except RateLimitedError as e:
                rate_error = rate_error or e
                logging.error(f"All transcript downloaders failed (rate-limited): {e}")
            except Exception as e4:
                logging.error(f"All transcript downloaders failed: {e4}")

            if rate_error:
                raise rate_error
            return None

        fetch_error = None
        try:
            transcript_text = await asyncio.wait_for(
                asyncio.to_thread(_fetch_transcript),
                timeout=180.0
            )
        except asyncio.TimeoutError:
            transcript_text = None
        except (NoSubtitlesError, AgeRestrictedError, PrivateVideoError, RateLimitedError) as e:
            fetch_error = e
            transcript_text = None
        except Exception as e:
            fetch_error = e
            transcript_text = None

        if not transcript_text:
            if isinstance(fetch_error, NoSubtitlesError):
                msg = (
                    "📭 <b>This video has no captions.</b>\n\n"
                    "YouTube doesn't provide subtitles for it, so automatic transcript extraction isn't possible.\n\n"
                    "👉 Please upload a <code>.txt</code> file with the transcript, "
                    "or copy and paste the script as a text message."
                )
            elif isinstance(fetch_error, AgeRestrictedError):
                msg = (
                    "🔞 <b>This video is age-restricted.</b>\n\n"
                    "It can't be accessed automatically without a logged-in account.\n\n"
                    "👉 Please upload a <code>.txt</code> file with the transcript, "
                    "or copy and paste the script as a text message."
                )
            elif isinstance(fetch_error, PrivateVideoError):
                msg = (
                    "🔒 <b>This video is private or unavailable.</b>\n\n"
                    "It can't be accessed because it's either private, deleted, or geo-blocked.\n\n"
                    "👉 Please upload a <code>.txt</code> file with the transcript, "
                    "or copy and paste the script as a text message."
                )
            else:
                # Rate-limited, timeout, or unknown transient error
                msg = (
                    "❌ <b>Could not download transcript automatically.</b>\n\n"
                    "YouTube is rate-limiting requests right now (temporary — usually resets within 1–2 hours).\n\n"
                    "👉 In the meantime:\n"
                    "1. Copy &amp; paste the transcript/script as a text message.\n"
                    "2. Or upload a <code>.txt</code> file with the transcript.\n\n"
                    "<i>In normal use (a few videos per day) this error won't appear.</i>"
                )
            await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
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
    
    print("Mediaroom bot is starting...")
    application.run_polling()


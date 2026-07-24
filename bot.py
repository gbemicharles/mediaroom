import os
import re
import logging
import subprocess
import uuid
import glob
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from youtube_transcript_api import YouTubeTranscriptApi
from config import check_env_vars, TELEGRAM_BOT_TOKEN, DEFAULT_PROMPT, get_webshare_proxies
from ai_services import generate_text_and_extract_prompt, generate_thumbnail, generate_intro_video

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

def download_transcript_api(video_url: str) -> str:
    """Downloads transcript using youtube-transcript-api, trying Webshare proxies in turn."""
    from youtube_transcript_api.proxies import GenericProxyConfig

    video_id = extract_video_id(video_url)
    if not video_id:
        raise Exception("Could not extract video ID from URL")

    proxy_urls = get_webshare_proxies()
    # Build list of configs to try: each proxy first, then no-proxy as last resort
    configs_to_try = [GenericProxyConfig(http_url=p, https_url=p) for p in proxy_urls] + [None]

    last_err = None
    for proxy_config in configs_to_try:
        try:
            api = YouTubeTranscriptApi(proxy_config=proxy_config)
            transcript_list = api.list(video_id)
            langs = [t.language_code for t in transcript_list]
            if not langs:
                raise Exception("No transcripts available for this video.")
            transcript = transcript_list.find_transcript(langs)
            transcript_data = transcript.fetch()
            return " ".join([entry.text for entry in transcript_data])
        except Exception as e:
            last_err = e
            continue

    raise Exception(f"youtube-transcript-api failed: {last_err}")

def download_transcript_invidious(video_url: str) -> str:
    """Downloads transcript via public Invidious instances (bypasses YouTube IP blocks)."""
    import requests as req
    video_id = extract_video_id(video_url)
    if not video_id:
        raise Exception("Could not extract video ID from URL")

    # Public Invidious instances to try in order
    instances = [
        "https://inv.nadeko.net",
        "https://invidious.nikkosphere.com",
        "https://iv.melmac.space",
        "https://invidious.privacydev.net",
        "https://yt.artemislena.eu",
        "https://invidious.perennialte.ch",
        "https://invidious.io.lol",
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
    """Downloads transcript using yt-dlp with Webshare proxy, cleans it, and returns the text."""
    unique_id = str(uuid.uuid4())
    temp_template = os.path.join(os.getcwd(), f"temp_{unique_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--write-auto-subs",
        "--write-subs",
        "--skip-download",
        "-o", temp_template,
    ]

    proxy_urls = get_webshare_proxies()
    if proxy_urls:
        cmd += ["--proxy", proxy_urls[0]]

    cmd.append(video_url)
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or str(e)
        raise Exception(f"yt-dlp failed to download subtitles: {error_msg}")
        
    # Find downloaded subtitle files
    pattern = os.path.join(os.getcwd(), f"temp_{unique_id}.*")
    downloaded_files = glob.glob(pattern)
    if not downloaded_files:
        raise Exception("No subtitle files could be found. This video may not have subtitles available.")
        
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

async def process_transcript(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, transcript_text: str, target_lang: str):
    try:
        prompt_to_use = user_prompts.get(user_id, DEFAULT_PROMPT)
        
        await context.bot.send_message(chat_id=chat_id, text=f"🧠 Translating transcript & generating assets in {target_lang}...")
        
        full_text, image_prompt, translated_transcript = generate_text_and_extract_prompt(transcript_text, prompt_to_use, target_lang)
        
        # Send translated transcript as a file
        if translated_transcript:
            filename = f"transcript_{target_lang}.txt"
            filepath = os.path.join(os.getcwd(), filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(translated_transcript)
            try:
                with open(filepath, "rb") as f:
                    await context.bot.send_document(
                        chat_id=chat_id,
                        document=f,
                        filename=filename,
                        caption=f"📝 Translated Transcript ({target_lang})"
                    )
            except Exception as e:
                logging.error(f"Error sending document: {e}")
                await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Failed to send translated transcript file: {e}")
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

        # Send generated text assets
        if len(full_text) > 4000:
            for i in range(0, len(full_text), 4000):
                await context.bot.send_message(chat_id=chat_id, text=full_text[i:i+4000])
        else:
            await context.bot.send_message(chat_id=chat_id, text=full_text)
            
        if image_prompt and image_prompt != "A generic YouTube thumbnail":
            await context.bot.send_message(chat_id=chat_id, text=f"🎨 Generating thumbnail based on prompt: \n`{image_prompt}`")
            image_url = generate_thumbnail(image_prompt)
            
            if image_url:
                await context.bot.send_photo(chat_id=chat_id, photo=image_url, caption="✅ Generated Thumbnail")
                
                await context.bot.send_message(chat_id=chat_id, text="🎬 Generating short intro video from the thumbnail (this might take a minute)...")
                video_url = generate_intro_video(image_url)
                
                if video_url:
                    await context.bot.send_video(chat_id=chat_id, video=video_url, caption="✅ Generated Intro Video")
                else:
                    await context.bot.send_message(chat_id=chat_id, text="❌ Failed to generate intro video.")
            else:
                await context.bot.send_message(chat_id=chat_id, text="❌ Failed to generate thumbnail.")
        else:
            await context.bot.send_message(chat_id=chat_id, text="⚠️ Could not extract a thumbnail prompt from the AI's response.")
            
    except Exception as e:
        logging.error(f"Error in processing: {e}")
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
        try:
            # Try youtube-transcript-api first
            transcript_text = download_transcript_api(video_url)
        except Exception as api_err:
            logging.info(f"youtube-transcript-api failed: {api_err}. Trying Invidious fallback...")
            try:
                # Fall back to Invidious (bypasses YouTube cloud IP blocks)
                transcript_text = download_transcript_invidious(video_url)
            except Exception as invidious_err:
                logging.info(f"Invidious failed: {invidious_err}. Trying yt-dlp fallback...")
                try:
                    # Last resort: yt-dlp
                    transcript_text = download_transcript_ytdlp(video_url)
                except Exception as ytdlp_err:
                    logging.error(f"All transcript downloaders failed.")
                    error_msg = (
                        "❌ Could not download transcript automatically.\n\n"
                        "YouTube is blocking requests from this server's IP address.\n\n"
                        "👉 What you can do:\n"
                        "1. Copy & paste the script/transcript directly as a text message to this bot.\n"
                        "2. Or upload a .txt file containing the script/transcript."
                    )
                    await context.bot.send_message(chat_id=chat_id, text=error_msg)
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


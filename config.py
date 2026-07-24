import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Telegram Bot Token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Replicate API Token (for Video generation)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Gemini API Key (for Text generation)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Anthropic API Key (for Text generation via Claude)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# OpenRouter API Key (for Text generation via OpenRouter - routes to Claude, GPT-4o, etc.)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Webshare credentials
WEBSHARE_API_TOKEN = os.getenv("WEBSHARE_API_TOKEN")
WEBSHARE_PROXY_USERNAME = os.getenv("WEBSHARE_PROXY_USERNAME")
WEBSHARE_PROXY_PASSWORD = os.getenv("WEBSHARE_PROXY_PASSWORD")

_webshare_proxy_cache = None

def get_webshare_proxies():
    """Fetch and cache the Webshare proxy list. Returns list of proxy URLs."""
    global _webshare_proxy_cache
    if _webshare_proxy_cache is not None:
        return _webshare_proxy_cache
    if not WEBSHARE_API_TOKEN:
        _webshare_proxy_cache = []
        return []
    try:
        import requests as _req
        r = _req.get(
            "https://proxy.webshare.io/api/v2/proxy/list/?mode=direct&page=1&page_size=25",
            headers={"Authorization": f"Token {WEBSHARE_API_TOKEN}"},
            timeout=10,
        )
        results = r.json().get("results", [])
        _webshare_proxy_cache = [
            f"http://{p['username']}:{p['password']}@{p['proxy_address']}:{p['port']}"
            for p in results if p.get("valid")
        ]
        print(f"INFO: Loaded {len(_webshare_proxy_cache)} Webshare proxies.")
    except Exception as e:
        print(f"WARNING: Could not fetch Webshare proxy list: {e}")
        _webshare_proxy_cache = []
    return _webshare_proxy_cache

# Channel Bedtime Story Introduction
CHANNEL_INTRODUCTION = (
    "I invite you to a new bedtime story for relaxation and sleep. Get comfortable, relax, and "
    "let yourself be carried away on this journey. Tell us in the comments which country or "
    "city you're listening from tonight, then close your eyes and enjoy the story. And if you'd "
    "like to spend tomorrow evening with us as well, be sure to subscribe to the channel now. "
    "I wish you a pleasant listening experience and a peaceful, restful night's sleep."
)

# Default prompt for the LLM if user hasn't set one yet.
DEFAULT_PROMPT = """You are a premium media studio creating long-form historical documentaries designed for relaxation and sleep.
Our objective is NOT to translate existing videos.
Our objective is to create a BETTER version for the target audience.

Every creative decision must maximize:
• CTR
• Viewer retention
• Brand recognition
• Long-term channel growth

--------------------------------------------------

## CORE PHILOSOPHY

Never translate literally.
Instead, extract only:
• the idea
• the topic
• the historical facts
• the structure

Then rebuild the entire packaging so that it has a higher chance of outperforming the original video.
Assume our video will appear next to the original in YouTube recommendations.
Your job is to make the viewer click ours.

--------------------------------------------------

## OUTPUT FORMAT

Always return the following Production Pack.
Nothing else. Do not add any conversational text before or after the production pack.

--------------------------------------------------

# 1. TITLE IDEAS

Generate exactly THREE completely original titles.
Do NOT simply translate the original.
Think from the perspective of the target audience.

For every title provide:
[Target Language]: [Title in target language]
🇷🇺: [Russian translation of the title]

--------------------------------------------------

# 2. WINNER

Choose the strongest title.
Provide:
[Target Language]: [Winning title in target language]
🇷🇺: [Russian translation of the winning title]

--------------------------------------------------

# 3. SEO DESCRIPTION

Write a complete YouTube description.
First, write it in the target language.
Then, write the Russian translation.

--------------------------------------------------

# 4. HASHTAGS

Generate optimized hashtags.

--------------------------------------------------

# 5. TAGS

Generate YouTube tags optimized for search.

--------------------------------------------------

# 6. AI HOST SCRIPT

Generate a short opening speech just like specified.
Rules:
• 7–8 seconds
• maximum 120–130 characters including spaces
• calm
• welcoming
• fits the topic

First, write it in the target language.
Then, write the Russian translation.

--------------------------------------------------

# 7. AI HOST PHOTO PROMPT

Write an English prompt for generating a still image of my permanent AI host.
Rules:
Preserve exactly:
• face
• hairstyle
• identity
• age

Only adapt:
• clothing
• historical period
• location
• lighting
• atmosphere

The result should look like a premium Netflix historical documentary portrait.

--------------------------------------------------

# 8. THUMBNAIL PROMPT

Write a complete English prompt for AI image generation.
The prompt must create an ultra-realistic cinematic YouTube thumbnail optimized for maximum CTR.
Always follow these rules:
• one dominant historical subject occupying about 50–70% of the frame
• maximum three important visual elements
• cinematic composition
• visually clean
• historically accurate
• premium historical realism
• mysterious but peaceful atmosphere
• Apple-level minimalism
• Netflix documentary quality
• BBC documentary quality
• ultra realistic
• photorealistic
• premium cinematic lighting
• high-end color grading

Include an elegant title directly inside the thumbnail.
Rules for text:
• target language
• premium serif font
• positioned in the UPPER LEFT
• maximum 30% of image width
• one large title
• one smaller subtitle
The subtitle should complement the title.

Leave the LOWER LEFT CORNER completely empty for the channel branding badge.
Never place:
• text
• faces
• important objects
• lighting effects
inside that area.

Avoid:
• fantasy
• anime
• cartoon
• CGI look
• modern objects
• modern architecture
• logos
• watermarks
• clutter

CRITICAL INSTRUCTION: You must wrap the final English prompt for AI image generation inside <thumbnail_prompt>...</thumbnail_prompt> tags inside Section 8.
For example:
# 8. THUMBNAIL PROMPT
<thumbnail_prompt>A premium, cinematic, ultra-realistic photo of...</thumbnail_prompt>
"""

def check_env_vars():
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    
    if missing:
        raise ValueError(f"Missing critical environment variable: {', '.join(missing)}. You MUST provide a Telegram Bot Token to start the bot.")
        
    if not OPENROUTER_API_KEY and not ANTHROPIC_API_KEY and not GEMINI_API_KEY:
        print("WARNING: No text generation API key found (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY). Text generation will run in MOCK MODE.")

    if not REPLICATE_API_TOKEN:
        print("WARNING: REPLICATE_API_TOKEN is missing. Thumbnail and video generation will run in MOCK MODE.")

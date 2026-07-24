# YouTube Automation Telegram Bot

A Telegram bot that acts as a YouTube content production assistant. Send it a transcript (or a YouTube link) and it generates titles, SEO descriptions, hashtags, tags, an AI host script, a DALL-E 3 thumbnail, and a short intro video.

## How to run

The bot runs with:
```
python bot.py
```

## Required secrets

| Secret | Required | Purpose |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | ✅ Yes | Bot won't start without this |
| `OPENAI_API_KEY` | Optional | DALL-E 3 thumbnail generation (mock mode if missing) |
| `REPLICATE_API_TOKEN` | Optional | Intro video generation via Stable Video Diffusion (mock mode if missing) |
| `GEMINI_API_KEY` | Optional | Text/transcript generation via Gemini 1.5 Flash; falls back to OpenAI GPT-4o if missing |

Set secrets via Replit's Secrets panel (not in a `.env` file).

## Bot usage

- `/start` — welcome message
- `/setprompt <text>` — override the default content generation prompt
- Send a YouTube URL → bot downloads the transcript automatically
- Send a `.txt` file (with optional YouTube link in the caption) → bot uses that as the transcript
- Send plain text → treated as the transcript directly

After receiving input, the bot asks you to pick a target language, then generates the full production pack.

## Mock mode

Without AI API keys the bot still runs: text output uses hardcoded mock responses and thumbnail/video generation returns placeholder URLs.

## User preferences

- Use existing project structure and stack; do not restructure without being asked.

"""
Tests for NoSubtitlesError detection and the Telegram handler's user-facing message.

Covers:
 - download_transcript_direct: missing captionTracks JSON key → NoSubtitlesError
 - download_transcript_direct: empty captionTracks list → NoSubtitlesError
 - download_transcript_invidious: empty captions list on every instance → NoSubtitlesError
 - language_callback handler: NoSubtitlesError → friendly "no captions" Telegram message
"""

import asyncio
import json
import sys
import os
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so we can import bot.py modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Minimal stubs for heavy deps that bot.py imports at module level
# ---------------------------------------------------------------------------
def _stub_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _ensure_stubs():
    # telegram stubs
    if "telegram" not in sys.modules:
        tg = _stub_module("telegram")
        tg.Update = MagicMock
        tg.InlineKeyboardButton = MagicMock
        tg.InlineKeyboardMarkup = MagicMock
        _stub_module("telegram.ext").ApplicationBuilder = MagicMock
        _stub_module("telegram.ext").ContextTypes = MagicMock
        _stub_module("telegram.ext").CommandHandler = MagicMock
        _stub_module("telegram.ext").MessageHandler = MagicMock
        _stub_module("telegram.ext").CallbackQueryHandler = MagicMock
        _stub_module("telegram.ext").filters = MagicMock()

    tg_ext = sys.modules.get("telegram.ext", _stub_module("telegram.ext"))
    for attr in ("ApplicationBuilder", "ContextTypes", "CommandHandler",
                 "MessageHandler", "CallbackQueryHandler", "filters"):
        if not hasattr(tg_ext, attr):
            setattr(tg_ext, attr, MagicMock())

    # config stub
    if "config" not in sys.modules:
        cfg = _stub_module("config")
        cfg.check_env_vars = MagicMock()
        cfg.TELEGRAM_BOT_TOKEN = "fake-token"
        cfg.DEFAULT_PROMPT = "default prompt"
        cfg.get_webshare_proxies = MagicMock(return_value=None)
        cfg.WEBSHARE_PROXY_USERNAME = ""
        cfg.WEBSHARE_PROXY_PASSWORD = ""

    # ai_services stub
    if "ai_services" not in sys.modules:
        ai = _stub_module("ai_services")
        ai.generate_text_and_extract_prompt = MagicMock()
        ai.generate_thumbnail = MagicMock()
        ai.generate_intro_video = MagicMock()
        ai.translate_chunk = MagicMock()
        ai.split_transcript_into_chunks = MagicMock(return_value=["chunk"])
        ai.split_into_paragraphs = MagicMock(return_value=["paragraph"])


_ensure_stubs()

import bot  # noqa: E402  (imported after stubs are in place)
from bot import (
    NoSubtitlesError,
    download_transcript_direct,
    download_transcript_invidious,
    language_callback,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_response(status_code: int = 200, text: str = "", json_data=None):
    """Return a MagicMock that mimics requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json = MagicMock(return_value=json_data)
    return resp


def _watch_page_html(caption_tracks_json: str) -> str:
    """Minimal watch-page HTML containing an embedded captionTracks JSON blob."""
    return f'"captionTracks":{caption_tracks_json},"some_other_key":1'


# ===========================================================================
# download_transcript_direct — unit tests
# ===========================================================================

class TestDownloadTranscriptDirect(unittest.TestCase):

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro) if asyncio.iscoroutine(coro) else coro

    @patch("bot.WEBSHARE_PROXY_USERNAME", "")
    @patch("bot.WEBSHARE_PROXY_PASSWORD", "")
    @patch("requests.get")
    def test_missing_caption_tracks_raises_no_subtitles_error(self, mock_get):
        """Watch-page HTML with NO captionTracks key → NoSubtitlesError."""
        watch_page = _make_response(text="<html>no caption tracks here</html>")
        mock_get.return_value = watch_page

        with self.assertRaises(NoSubtitlesError):
            download_transcript_direct("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("bot.WEBSHARE_PROXY_USERNAME", "")
    @patch("bot.WEBSHARE_PROXY_PASSWORD", "")
    @patch("requests.get")
    def test_empty_caption_tracks_raises_no_subtitles_error(self, mock_get):
        """Watch-page HTML with empty captionTracks array → NoSubtitlesError."""
        html = _watch_page_html("[]")
        watch_page = _make_response(text=html)
        mock_get.return_value = watch_page

        with self.assertRaises(NoSubtitlesError):
            download_transcript_direct("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("bot.WEBSHARE_PROXY_USERNAME", "")
    @patch("bot.WEBSHARE_PROXY_PASSWORD", "")
    @patch("requests.get")
    def test_valid_caption_tracks_returns_transcript(self, mock_get):
        """Watch-page with valid captionTracks + valid XML → transcript string."""
        track = {"languageCode": "en", "baseUrl": "https://example.com/captions"}
        html = _watch_page_html(json.dumps([track]))
        watch_page = _make_response(text=html)

        xml_body = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<transcript>"
            '<text start="0" dur="1">Hello world</text>'
            '<text start="1" dur="1">This is a test</text>'
            "</transcript>"
        )
        caption_response = _make_response(text=xml_body)

        mock_get.side_effect = [watch_page, caption_response]

        result = download_transcript_direct("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIn("Hello world", result)
        self.assertIn("This is a test", result)


# ===========================================================================
# download_transcript_invidious — unit tests
# ===========================================================================

class TestDownloadTranscriptInvidious(unittest.TestCase):

    @patch("requests.get")
    def test_empty_captions_list_raises_no_subtitles_error(self, mock_get):
        """All Invidious instances return empty captions list → NoSubtitlesError."""
        resp = _make_response(json_data={"captions": []})
        mock_get.return_value = resp

        with self.assertRaises(NoSubtitlesError):
            download_transcript_invidious("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("requests.get")
    def test_missing_captions_key_raises_no_subtitles_error(self, mock_get):
        """All Invidious instances return JSON with no 'captions' key → NoSubtitlesError."""
        resp = _make_response(json_data={})
        mock_get.return_value = resp

        with self.assertRaises(NoSubtitlesError):
            download_transcript_invidious("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("requests.get")
    def test_valid_captions_returns_transcript(self, mock_get):
        """Invidious returns caption track with valid XML content → transcript string."""
        captions_resp = _make_response(json_data={
            "captions": [
                {"language_code": "en", "url": "/api/v1/captions/dQw4w9WgXcQ?label=English"}
            ]
        })
        xml_body = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<transcript>"
            '<p start="0" dur="1000">First line</p>'
            '<p start="1000" dur="1000">Second line</p>'
            "</transcript>"
        )
        subtitle_resp = _make_response(text=xml_body)

        mock_get.side_effect = [captions_resp, subtitle_resp]

        result = download_transcript_invidious("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIn("First line", result)
        self.assertIn("Second line", result)


# ===========================================================================
# language_callback — integration-level test for the Telegram handler
# ===========================================================================

class TestLanguageCallbackNoSubtitles(unittest.IsolatedAsyncioTestCase):
    """
    Verifies that when _fetch_transcript raises NoSubtitlesError, the Telegram
    handler sends the friendly 'no captions' message instead of a generic error.
    """

    async def _run_callback_with_fetch_error(self, error: Exception) -> str:
        """
        Drive language_callback end-to-end with a stubbed update/context, patching
        asyncio.to_thread so _fetch_transcript immediately raises `error`.
        Returns the text of the first send_message call made after the error.
        """
        # Build a minimal fake CallbackQuery / Update
        query = MagicMock()
        query.data = "lang_English"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        update.effective_user.id = 42
        update.effective_chat.id = 99

        sent_messages = []

        bot_mock = MagicMock()
        async def _send_message(chat_id, text, parse_mode=None):
            sent_messages.append(text)
        bot_mock.send_message = AsyncMock(side_effect=_send_message)

        context = MagicMock()
        context.bot = bot_mock
        context.user_data = {
            "pending_video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "pending_transcript": None,
            "pending_caption": "",
        }

        # Patch asyncio.to_thread so the inner _fetch_transcript raises our error
        async def _fake_to_thread(fn, *args, **kwargs):
            raise error

        with patch("bot.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("bot.asyncio.wait_for", side_effect=_fake_to_thread):
            # wait_for wraps to_thread, so patch it directly to raise the error
            async def _fake_wait_for(coro, timeout):
                raise error
            with patch("bot.asyncio.wait_for", side_effect=_fake_wait_for):
                await language_callback(update, context)

        return sent_messages

    async def test_no_subtitles_error_sends_friendly_message(self):
        msgs = await self._run_callback_with_fetch_error(
            NoSubtitlesError("This video has no captions available.")
        )
        self.assertTrue(
            msgs,
            "Expected at least one send_message call after NoSubtitlesError"
        )
        combined = " ".join(msgs)
        # Must contain the friendly no-captions indicator
        self.assertIn("no captions", combined.lower(),
                      f"Expected 'no captions' in message, got: {combined!r}")
        # Must NOT be a raw exception dump
        self.assertNotIn("NoSubtitlesError", combined,
                         "Handler should not expose exception class name to user")

    async def test_no_subtitles_error_message_contains_upload_suggestion(self):
        """The friendly message should suggest uploading a .txt file."""
        msgs = await self._run_callback_with_fetch_error(
            NoSubtitlesError("This video has no captions available.")
        )
        combined = " ".join(msgs)
        self.assertTrue(
            ".txt" in combined or "transcript" in combined.lower(),
            f"Expected upload suggestion in message, got: {combined!r}"
        )


if __name__ == "__main__":
    unittest.main()

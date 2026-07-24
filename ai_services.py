import re
import logging
import replicate
import anthropic
from openai import OpenAI
import google.generativeai as genai
from config import OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, REPLICATE_API_TOKEN, GEMINI_API_KEY, CHANNEL_INTRODUCTION

# Initialize clients safely
# OpenRouter uses the OpenAI SDK format with a different base URL
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
) if OPENROUTER_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
# Replicate automatically picks up REPLICATE_API_TOKEN from env vars

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
            f"You must perform the following tasks:\n"
            f"1. Generate the Production Pack exactly as structured. Use {target_lang} for target language titles, winning title, SEO description, and AI Host Script, and also provide the Russian translation (labeled 🇷🇺:) for those same sections.\n"
            f"2. Literally and professionally translate the following channel introduction into {target_lang}, "
            f"and wrap it in <translated_intro>...</translated_intro> tags. Do not creatively rewrite this introduction. The introduction is:\n"
            f"\"{CHANNEL_INTRODUCTION}\"\n\n"
            f"3. Completely rebuild and rewrite the input transcript/content into a premium historical documentary script in {target_lang} designed for relaxation and sleep. "
            f"You must wrap this rewritten transcript in <translated_transcript>...</translated_transcript> tags. Follow these rules for this script rewrite:\n"
            f"   - Never translate literally. Instead, extract only the idea, the topic, the historical facts, and the structure of the input, and write a completely new, better script in {target_lang}.\n"
            f"   - Keep the narrative logical, calm, and natural, maintaining historical facts (names, titles, cities, countries, dates, times) but rewriting all phrasings to make it engaging and soothing.\n"
            f"   - Convert all numerals (numbers, digits) into full words in {target_lang}.\n"
            f"   - Output the rewritten script as one single continuous block of text inside the <translated_transcript> tags, without breaking it into paragraphs or adding additional comments.\n"
            f"   - Automatically check for and remove advertisements, sponsored content, promotions, and mentions of other channels.\n\n"
            f"Do NOT translate the AI host photo prompt or the thumbnail prompt; they must remain in English.\n"
            f"Remember to wrap the image generation prompt in <thumbnail_prompt>...</thumbnail_prompt> inside Section 8."
        )
    else:
        lang_instruction = (
            f"\n\nCRITICAL INSTRUCTION FOR TRANSLATION AND REWRITING:\n"
            f"You must perform two main tasks regarding the transcript and the channel's introduction:\n"
            f"1. Literally and professionally translate the following channel introduction into {target_lang}, "
            f"and wrap it in <translated_intro>...</translated_intro> tags. Do not creatively rewrite this introduction;\n"
            f"keep it as a faithful translation. The introduction is:\n"
            f"\"{CHANNEL_INTRODUCTION}\"\n\n"
            f"2. Translate, revise, and clean the transcript of the video into {target_lang}. "
            f"You must wrap this rewritten transcript in <translated_transcript>...</translated_transcript> tags.\n"
            f"Follow these strict rules for the transcript rewrite:\n"
            f"- Make the subtitles look normal and natural.\n"
            f"- Significantly revise the text to make it as different from the original phrasing as possible. Make it a completely new script.\n"
            f"- Keep the narrative logical and natural while maintaining the original storyline and meaning.\n"
            f"- Change words to synonyms when necessary, but ensure they are appropriate for the context.\n"
            f"- You may add appropriate new text, as long as the core meaning is preserved.\n"
            f"- You can rearrange/change semantic blocks of text in places, but ensure the narrative logic is preserved.\n"
            f"- Do NOT change or delete names, titles, cities, countries, dates, times, or factual details that are important to the context.\n"
            f"- Provide correct grammar and punctuation according to the rules of {target_lang}.\n"
            f"- Convert all numerals (numbers, digits) into full words in {target_lang} that accurately match their original meaning.\n"
            f"- Output the rewritten transcript as one single continuous block of text inside the <translated_transcript> tags, without breaking it into paragraphs or adding additional comments.\n"
            f"- Automatically check the text for advertisements, sponsored content, promotions, and mentions of other channels, and completely remove them.\n\n"
            f"Also, generate the YouTube titles, description, tags, and hashtags in {target_lang} outside of the tags."
        )

    full_text = ""

    if openrouter_client:
        response = openrouter_client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            max_tokens=4096,
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
    elif openai_client:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt + lang_instruction},
                {"role": "user", "content": f"Here is the transcript of the video:\n\n{transcript}"}
            ],
            temperature=0.7
        )
        full_text = response.choices[0].message.content
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
# 2. WINNER
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
    """Calls FLUX Schnell via Replicate to generate a thumbnail and returns the URL."""
    if not REPLICATE_API_TOKEN:
        return "https://placehold.co/1280x720/png?text=Mock+Thumbnail"

    try:
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": image_prompt,
                "num_outputs": 1,
                "aspect_ratio": "16:9",
                "output_format": "jpg",
                "output_quality": 90,
                "go_fast": True,
            }
        )
        # Output is a list of URLs or FileOutput objects
        result = output[0] if isinstance(output, list) else output
        if hasattr(result, 'url'):
            return str(result.url)
        return str(result)
    except Exception as e:
        print(f"Error generating thumbnail: {e}")
        return ""

def generate_intro_video(image_url: str) -> str:
    """Calls Replicate to generate a short intro video based on the thumbnail."""
    if not REPLICATE_API_TOKEN:
        return "https://www.w3schools.com/html/mov_bbb.mp4"

    try:
        output = replicate.run(
            "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
            input={
                "cond_aug": 0.02,
                "decoding_t": 7,
                "input_image": image_url,
                "video_length": "14_frames_with_svd_xt",
                "sizing_strategy": "maintain_aspect_ratio",
                "motion_bucket_id": 127,
                "frames_per_second": 6
            }
        )
        return output
    except Exception as e:
        print(f"Error generating intro video: {e}")
        return ""

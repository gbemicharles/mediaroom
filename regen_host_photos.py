"""
Regenerate all 12 host photos with cinematic landscape backgrounds.
Run once: python regen_host_photos.py
"""
import os
import sys
import requests
import concurrent.futures
import fal_client

FAL_API_KEY = os.environ.get("FAL_KEY", "")
if not FAL_API_KEY:
    sys.exit("FAL_KEY not set")
os.environ["FAL_KEY"] = FAL_API_KEY

BASE = """Photorealistic cinematic portrait, full body shot, {subject} wearing {attire},
standing confidently on {location}, {landscape_detail},
golden hour warm sunlight, atmospheric mist in the valley below, bokeh background,
dramatic depth of field, 35mm film, professional cinematography, 
Terrence Malick lighting, ultra-detailed, 8K, no text, single person only"""

SPECS = {
    "ro": {
        "subject": "a Romanian nobleman in his 30s, dark hair, warm complexion",
        "attire": "19th century Romanian boyar attire — embroidered velvet coat, white cravat",
        "location": "a rocky cliff edge in the Carpathian Mountains",
        "landscape_detail": "misty forested valleys and medieval Transylvanian castle ruins below, winding river",
    },
    "en": {
        "subject": "a Victorian English gentleman in his 40s, strong jaw",
        "attire": "dark Victorian frock coat, white shirt, silk cravat",
        "location": "dramatic chalk white coastal cliffs overlooking the sea",
        "landscape_detail": "deep blue English Channel, sea mist, copper sunset sky",
    },
    "es": {
        "subject": "a Spanish nobleman in his 30s, dark eyes, confident expression",
        "attire": "19th century Spanish hidalgo coat with gold embroidery",
        "location": "ancient Moorish castle battlements in Andalusia",
        "landscape_detail": "orange groves and olive trees stretching to misty mountains, warm amber light",
    },
    "fr": {
        "subject": "a French intellectual in his late 30s, refined features",
        "attire": "19th century French redingote coat, ivory cravat",
        "location": "hilltop overlooking the Loire Valley",
        "landscape_detail": "river winding through green valley, Renaissance château below, golden-hour haze",
    },
    "de": {
        "subject": "a German scholar in his 40s, serious expression",
        "attire": "19th century German Biedermeier frock coat, dark waistcoat",
        "location": "alpine meadow on a mountain ridge",
        "landscape_detail": "snow-capped peaks behind, misty valleys, pine forests, low golden sun",
    },
    "pt": {
        "subject": "a Portuguese nobleman in his 30s, expressive dark eyes",
        "attire": "19th century Portuguese nobleman coat, velvet collar",
        "location": "rugged Atlantic sea cliff at Cabo da Roca",
        "landscape_detail": "crashing ocean waves below, dramatic sunset clouds, golden light on the water",
    },
    "it": {
        "subject": "an Italian Renaissance scholar in his 30s, classical features",
        "attire": "19th century Italian aristocrat coat, silk vest",
        "location": "hilltop terrace overlooking Tuscan countryside",
        "landscape_detail": "cypress-lined roads, rolling hills, vineyards, warm amber evening light",
    },
    "ru": {
        "subject": "a Russian nobleman in his 40s, strong Slavic features",
        "attire": "19th century Russian Imperial uniform coat, epaulettes",
        "location": "birch forest hill in the Russian countryside",
        "landscape_detail": "golden autumn birch trees, misty river below, low winter sun rays",
    },
    "pl": {
        "subject": "a Polish nobleman in his 30s, proud bearing",
        "attire": "19th century Polish kontusz coat, traditional sash",
        "location": "castle hill overlooking the Vistula River",
        "landscape_detail": "medieval town below, vast plains beyond, dramatic sunset clouds",
    },
    "zh": {
        "subject": "a Chinese scholar-official in his 40s, wise expression",
        "attire": "Qing dynasty silk scholar robe, mandarin collar",
        "location": "mist-shrouded mountain peak among the Huangshan peaks",
        "landscape_detail": "pine trees emerging from clouds, dramatic karst mountain formations, ethereal morning mist",
    },
    "ja": {
        "subject": "a Japanese nobleman in his 30s, composed expression",
        "attire": "Meiji-era hakama and haori jacket, traditional obi",
        "location": "zen garden terrace overlooking Mount Fuji",
        "landscape_detail": "cherry blossoms in foreground, snow-capped Fuji in distance, pink-golden dawn light",
    },
    "tr": {
        "subject": "an Ottoman official in his 40s, dignified bearing",
        "attire": "19th century Ottoman Tanzimat-era frock coat and fez",
        "location": "hilltop palace terrace overlooking the Bosphorus",
        "landscape_detail": "Istanbul skyline with minarets, golden sunset reflecting on the strait, atmospheric haze",
    },
}


def generate_one(lang: str, spec: dict) -> tuple[str, bool]:
    prompt = BASE.format(**spec)
    out_path = f"host_photos/{lang}.jpg"
    try:
        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1.1",
            arguments={
                "prompt": prompt,
                "image_size": "landscape_16_9",
                "output_format": "jpeg",
                "output_quality": 95,
                "safety_tolerance": 6,
                "num_images": 1,
            },
        )
        url = result["images"][0]["url"]
        img_data = requests.get(url, timeout=60).content
        with open(out_path, "wb") as f:
            f.write(img_data)
        print(f"  ✓ {lang}: saved {len(img_data)//1024}KB → {out_path}")
        return lang, True
    except Exception as e:
        print(f"  ✗ {lang}: FAILED — {e}")
        return lang, False


print(f"Regenerating {len(SPECS)} host photos in parallel…")
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    futures = {pool.submit(generate_one, lang, spec): lang for lang, spec in SPECS.items()}
    results = {lang: ok for fut in concurrent.futures.as_completed(futures) for lang, ok in [fut.result()]}

ok = sum(results.values())
print(f"\nDone: {ok}/{len(SPECS)} succeeded.")

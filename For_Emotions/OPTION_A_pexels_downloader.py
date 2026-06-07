"""
OPTION A — Pexels API Image Downloader
=======================================
Downloads images for all 27 emotions into their respective folders.

SETUP:
1. Get a FREE Pexels API key at: https://www.pexels.com/api/
2. Replace YOUR_PEXELS_API_KEY below with your key
3. Run: python3 OPTION_A_pexels_downloader.py

Pexels free tier: 200 requests/hour, 20,000/month — more than enough.
All Pexels images are free for use including AI training datasets.
"""

import os
import time
import requests
from pathlib import Path

# ============================================================
# CONFIGURATION — Edit these
# ============================================================
PEXELS_API_KEY = "ZB5dDjQFlbwAQzKLajgt8rUGIB0H1f6wcsserc6HHqepTZffvQox8KYo"   # Get free at pexels.com/api
IMAGES_PER_EMOTION = 30                   # How many images per folder (max 80)
BASE_FOLDER = Path(__file__).parent       # Same folder as this script
DELAY_BETWEEN_REQUESTS = 0.5             # Seconds between API calls (be polite)
# ============================================================

EMOTIONS = {
    "Admiration":              ["person admiring someone", "crowd applauding", "respectful gaze"],
    "Adoration":               ["parent adoring baby", "person hugging pet lovingly", "tender love moment"],
    "Aesthetic_Appreciation":  ["person appreciating artwork", "man watching sunset", "woman admiring architecture"],
    "Amusement":               ["person laughing joke", "friends amused together", "child giggling happy"],
    "Anger":                   ["angry person face", "man shouting rage", "furious expression closeup"],
    "Anxiety":                 ["anxious nervous person", "worried person face", "stressed anxious expression"],
    "Awe":                     ["person awestruck amazed", "jaw drop amazement face", "crowd watching spectacle awe"],
    "Awkwardness":             ["awkward social situation", "embarrassed uncomfortable person", "cringe awkward moment"],
    "Boredom":                 ["bored person face", "student bored class", "person staring blankly"],
    "Calmness":                ["calm serene person face", "woman meditating peaceful", "relaxed calm expression"],
    "Confusion":               ["confused puzzled person", "man scratching head confused", "puzzled expression face"],
    "Craving":                 ["person craving food", "longing desire face", "hungry person staring food"],
    "Disgust":                 ["disgusted face expression", "person recoiling disgust", "nose wrinkle disgust"],
    "Empathetic_Pain":         ["person comforting friend", "empathy compassionate face", "caring friend sad moment"],
    "Entrancement":            ["person mesmerized transfixed", "child entranced magic", "person absorbed focused"],
    "Excitement":              ["excited person jumping happy", "thrilled expression face", "fan excited sports"],
    "Fear":                    ["scared fearful person", "frightened face expression", "afraid wide eyes person"],
    "Horror":                  ["horrified person expression", "shocked horrified face", "extreme fear horror face"],
    "Interest":                ["interested curious person", "attentive face leaning forward", "intrigued expression person"],
    "Joy":                     ["joyful happy person beaming", "pure joy laughing child", "overjoyed celebration person"],
    "Nostalgia":               ["nostalgic person old photos", "wistful reminiscing expression", "person faraway look memory"],
    "Relief":                  ["relieved person exhaling", "relief expression after stress", "sigh of relief face"],
    "Romance":                 ["romantic couple tender moment", "romantic longing gaze", "affectionate couple expression"],
    "Sadness":                 ["sad person crying face", "grief sadness expression", "downcast sad eyes person"],
    "Satisfaction":            ["satisfied content person", "accomplished pleased expression", "satisfied smile after work"],
    "Sexual_Desire":           ["romantic attraction expression", "flirtatious subtle expression", "longing attraction face"],
    "Surprise":                ["surprised shocked person", "wide eyes open mouth surprise", "surprise reaction face"],
}

def download_image(url: str, filepath: Path) -> bool:
    """Download a single image from URL to filepath."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"    ✗ Failed to download {url}: {e}")
        return False


def search_pexels(query: str, per_page: int = 15, page: int = 1) -> list:
    """Search Pexels for images matching query."""
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
        "orientation": "square",
    }
    try:
        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("photos", [])
    except Exception as e:
        print(f"    ✗ Pexels API error: {e}")
        return []


def download_emotion(emotion: str, queries: list, target_count: int):
    """Download images for one emotion using multiple queries."""
    folder = BASE_FOLDER / emotion
    folder.mkdir(exist_ok=True)

    downloaded = 0
    img_index = 1
    queries_used = 0

    print(f"\n📁 {emotion}")

    for query in queries:
        if downloaded >= target_count:
            break

        needed = target_count - downloaded
        per_page = min(needed, 15)

        print(f"  🔍 Query: '{query}' ({per_page} images)")
        photos = search_pexels(query, per_page=per_page)
        queries_used += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)

        for photo in photos:
            if downloaded >= target_count:
                break

            img_url = photo["src"]["large"]  # Good quality, not huge file
            ext = "jpg"
            filename = folder / f"{emotion}_{img_index:03d}.{ext}"

            if download_image(img_url, filename):
                downloaded += 1
                img_index += 1
                print(f"    ✓ {filename.name} — credit: {photo.get('photographer', 'unknown')}")
            
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"  ✅ {downloaded}/{target_count} images downloaded for {emotion}")
    return downloaded


def main():
    print("=" * 60)
    print("  Emotion Image Downloader — Pexels API")
    print("=" * 60)

    if PEXELS_API_KEY == "YOUR_PEXELS_API_KEY":
        print("\n❌ ERROR: Please set your Pexels API key in the script.")
        print("   Get one free at: https://www.pexels.com/api/")
        return

    total_downloaded = 0
    total_emotions = len(EMOTIONS)

    print(f"\n📋 Downloading {IMAGES_PER_EMOTION} images × {total_emotions} emotions")
    print(f"   = ~{IMAGES_PER_EMOTION * total_emotions} total images\n")

    for i, (emotion, queries) in enumerate(EMOTIONS.items(), 1):
        print(f"[{i}/{total_emotions}]", end=" ")
        count = download_emotion(emotion, queries, IMAGES_PER_EMOTION)
        total_downloaded += count

    print("\n" + "=" * 60)
    print(f"  ✅ DONE! {total_downloaded} images downloaded across {total_emotions} emotions")
    print(f"  📁 Saved to: {BASE_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()

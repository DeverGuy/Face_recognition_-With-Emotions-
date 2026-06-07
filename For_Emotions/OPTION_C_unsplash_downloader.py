"""
OPTION C — Unsplash API Image Downloader (with resume support)
==============================================================
Downloads images for all 27 emotions with smart rate limiting,
progress tracking, and resume capability if interrupted.

SETUP:
1. Register a FREE Unsplash developer app at: https://unsplash.com/developers
2. Copy your "Access Key" (not secret key)
3. Replace YOUR_UNSPLASH_ACCESS_KEY below
4. Run: python3 OPTION_C_unsplash_downloader.py

Unsplash free tier: 50 requests/hour — script auto-throttles to stay safe.
Unsplash license: Free for any use including ML training datasets.
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION — Edit these
# ============================================================
UNSPLASH_ACCESS_KEY = "YOUR_UNSPLASH_ACCESS_KEY"  # unsplash.com/developers
IMAGES_PER_EMOTION  = 20       # Keep at 20 to stay within hourly limit
BASE_FOLDER         = Path(__file__).parent
PROGRESS_FILE       = BASE_FOLDER / ".download_progress.json"
DELAY_SECONDS       = 1.5      # Seconds between requests (50/hr limit = 72s safe buffer)
# ============================================================

EMOTIONS_QUERIES = {
    "Admiration":              "person looking admiringly",
    "Adoration":               "parent adoring baby tender",
    "Aesthetic_Appreciation":  "person appreciating art museum",
    "Amusement":               "person laughing amused",
    "Anger":                   "angry person face expression",
    "Anxiety":                 "anxious worried nervous person",
    "Awe":                     "person awestruck amazed wonder",
    "Awkwardness":             "awkward uncomfortable social",
    "Boredom":                 "bored person face disinterested",
    "Calmness":                "calm serene peaceful person",
    "Confusion":               "confused puzzled person thinking",
    "Craving":                 "person craving desire longing",
    "Disgust":                 "disgust expression face",
    "Empathetic_Pain":         "empathy compassion comforting friend",
    "Entrancement":            "person mesmerized transfixed absorbed",
    "Excitement":              "excited thrilled happy person",
    "Fear":                    "scared fearful frightened face",
    "Horror":                  "horrified shocked extreme fear",
    "Interest":                "curious interested attentive person",
    "Joy":                     "joyful happy beaming person",
    "Nostalgia":               "nostalgic wistful reminiscing person",
    "Relief":                  "relieved person exhale sigh",
    "Romance":                 "romantic couple tender moment",
    "Sadness":                 "sad grief sadness person crying",
    "Satisfaction":            "satisfied content accomplished person",
    "Sexual_Desire":           "romantic longing attraction subtle",
    "Surprise":                "surprised shocked amazed face",
}


def load_progress() -> dict:
    """Load download progress from file (for resume support)."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    """Save download progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def search_unsplash(query: str, page: int = 1, per_page: int = 20) -> list:
    """Search Unsplash for photos."""
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {
        "query": query,
        "page": page,
        "per_page": per_page,
        "orientation": "squarish",
        "content_filter": "high",  # Safe content only
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print("  ⚠️  Rate limit hit. Waiting 70 seconds...")
            time.sleep(70)
            return []
        print(f"  ✗ HTTP Error: {e}")
        return []
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


def download_image(url: str, filepath: Path, attribution: str) -> bool:
    """Download image and save attribution."""
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(r.content)
        # Save attribution alongside image (Unsplash requires attribution)
        attr_file = filepath.with_suffix('.txt')
        with open(attr_file, 'w') as f:
            f.write(f"Photo by {attribution} on Unsplash\nURL: {url}\n")
        return True
    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return False


def download_emotion(emotion: str, query: str, target: int, progress: dict) -> int:
    """Download images for one emotion, resuming if interrupted."""
    folder = BASE_FOLDER / emotion
    folder.mkdir(exist_ok=True)

    already_done = progress.get(emotion, 0)
    if already_done >= target:
        print(f"  ✅ {emotion}: Already complete ({already_done} images)")
        return already_done

    downloaded = already_done
    page = (downloaded // 20) + 1
    img_index = downloaded + 1

    print(f"\n📁 {emotion} (resuming from {downloaded})")
    print(f"   Query: '{query}'")

    while downloaded < target:
        photos = search_unsplash(query, page=page, per_page=20)
        time.sleep(DELAY_SECONDS)

        if not photos:
            print(f"  ⚠️  No more results for '{query}' at page {page}")
            break

        for photo in photos:
            if downloaded >= target:
                break

            img_url = photo["urls"]["regular"]  # ~1080px wide
            photographer = photo["user"]["name"]
            filename = folder / f"{emotion}_{img_index:03d}.jpg"

            if download_image(img_url, filename, photographer):
                downloaded += 1
                img_index += 1
                print(f"    [{downloaded}/{target}] ✓ {filename.name} — {photographer}")
            
            time.sleep(DELAY_SECONDS)

        page += 1

    progress[emotion] = downloaded
    save_progress(progress)
    return downloaded


def print_summary(progress: dict, start_time: float):
    """Print final summary."""
    total = sum(progress.values())
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    for emotion, count in progress.items():
        status = "✅" if count >= IMAGES_PER_EMOTION else "⚠️ "
        print(f"  {status} {emotion}: {count} images")
    print("-" * 60)
    print(f"  Total: {total} images | Time: {minutes}m {seconds}s")
    print(f"  Saved to: {BASE_FOLDER}")
    print("=" * 60)

    if total < len(EMOTIONS_QUERIES) * IMAGES_PER_EMOTION:
        print("\n  ℹ️  Some folders incomplete. Run script again to resume.")
        print(f"     Progress saved to: {PROGRESS_FILE}")


def main():
    print("=" * 60)
    print("  Emotion Image Downloader — Unsplash API (with resume)")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if UNSPLASH_ACCESS_KEY == "YOUR_UNSPLASH_ACCESS_KEY":
        print("\n❌ ERROR: Set your Unsplash Access Key in the script.")
        print("   Get one free at: https://unsplash.com/developers")
        return

    progress = load_progress()
    start_time = time.time()

    total_emotions = len(EMOTIONS_QUERIES)
    total_target = total_emotions * IMAGES_PER_EMOTION

    print(f"\n  Plan: {IMAGES_PER_EMOTION} images × {total_emotions} emotions = {total_target} total")
    print(f"  Rate: {DELAY_SECONDS}s delay → ~{int(total_target * DELAY_SECONDS / 60)} minutes estimated")
    print(f"  ⚠️  Unsplash limit: 50 requests/hour. Script auto-throttles.\n")

    for i, (emotion, query) in enumerate(EMOTIONS_QUERIES.items(), 1):
        print(f"\n[{i}/{total_emotions}]", end=" ")
        download_emotion(emotion, query, IMAGES_PER_EMOTION, progress)

    print_summary(progress, start_time)


if __name__ == "__main__":
    main()

# 🤖 For_Emotions — Humanoid Robot AI Training Dataset

## Overview
This folder contains **27 emotion categories** for training a Humanoid Robot AI Model to recognize and understand human emotional expressions.

Based on the **27-emotion framework** by Cowen & Keltner (2017), UC Berkeley.

---

## 📁 Folder Structure
```
For_Emotions/
├── Admiration/          ← README.md + images
├── Adoration/
├── Aesthetic_Appreciation/
├── Amusement/
├── Anger/
├── Anxiety/
├── Awe/
├── Awkwardness/
├── Boredom/
├── Calmness/
├── Confusion/
├── Craving/
├── Disgust/
├── Empathetic_Pain/
├── Entrancement/
├── Excitement/
├── Fear/
├── Horror/
├── Interest/
├── Joy/
├── Nostalgia/
├── Relief/
├── Romance/
├── Sadness/
├── Satisfaction/
├── Sexual_Desire/
├── Surprise/
├── OPTION_A_pexels_downloader.py   ← Auto-downloader (Pexels)
├── OPTION_C_unsplash_downloader.py ← Auto-downloader (Unsplash, with resume)
└── README.md                        ← This file
```

---

## 🚀 Three Ways to Get Images

### OPTION A — Pexels API Auto-Downloader
**File:** `OPTION_A_pexels_downloader.py`

| Feature | Detail |
|---------|--------|
| API | Pexels (free) |
| Rate limit | 200 requests/hour |
| Sign-up | https://www.pexels.com/api/ |
| License | Free for all uses including ML |
| Speed | Fast (~30 mins for 30 images × 27 emotions) |

**Steps:**
1. Go to https://www.pexels.com/api/ → Sign up → Copy API key
2. Open `OPTION_A_pexels_downloader.py`
3. Replace `YOUR_PEXELS_API_KEY` with your key
4. Run: `python3 OPTION_A_pexels_downloader.py`

---

### OPTION B — Manual Search Guides
**Location:** Each emotion folder has a `README.md` with:
- Visual cues to look for
- 5 curated search queries
- Direct links to professional datasets
- Image quality guidelines

Use these to manually search and download from:
- [Pexels](https://www.pexels.com) — Free, no sign-up needed
- [Unsplash](https://unsplash.com) — Free, high quality
- [Getty Images](https://www.gettyimages.com) — Paid, professional

---

### OPTION C — Unsplash API Auto-Downloader (Recommended ⭐)
**File:** `OPTION_C_unsplash_downloader.py`

| Feature | Detail |
|---------|--------|
| API | Unsplash (free) |
| Rate limit | 50 requests/hour |
| Sign-up | https://unsplash.com/developers |
| License | Free for all uses including ML |
| Extra | Resume if interrupted, attribution saved |

**Steps:**
1. Go to https://unsplash.com/developers → New App → Copy Access Key
2. Open `OPTION_C_unsplash_downloader.py`
3. Replace `YOUR_UNSPLASH_ACCESS_KEY` with your key
4. Run: `python3 OPTION_C_unsplash_downloader.py`
5. If interrupted, just run again — it resumes automatically ✅

---

## 🏆 Recommended Pre-Built Datasets (Best for AI Training)

These datasets are purpose-built for emotion recognition and far better than web images:

| Dataset | Emotions | Size | URL |
|---------|----------|------|-----|
| **AffectNet** | 8 basic + 11 compound | 1M+ images | http://mohammadmahoor.com/affectnet/ |
| **FER2013** | 7 basic emotions | 35,887 images | https://kaggle.com/datasets/msambare/fer2013 |
| **RAF-DB** | 7 basic + 12 compound | 30,000 images | http://www.whdeng.cn/RAF/model1.html |
| **EMOTIC** | 26 categories (body+context) | 23,571 images | https://s3.amazonaws.com/emotic-dataset |
| **CK+** (Extended Cohn-Kanade) | 8 emotions, posed | 10,000+ images | https://kaggle.com/datasets/shawon10/ckplus |
| **JAFFE** | 7 emotions, Japanese female | 213 images | https://zenodo.org/record/3451524 |
| **Aff-Wild2** | Valence + arousal + 8 emotions | 558 videos | https://ibug.doc.ic.ac.uk/resources/aff-wild2/ |
| **MELD** | 7 emotions, multi-modal | Dialogue videos | https://github.com/SenticNet/MELD |

> **Tip for Humanoid Robots:** EMOTIC is especially valuable because it captures emotions from full-body context (not just face), which is critical for robots operating in real environments.

---

## 📐 Image Standards for Training

| Parameter | Recommendation |
|-----------|----------------|
| Resolution | 224×224 minimum (512×512+ preferred) |
| Format | JPG or PNG |
| Color | RGB (convert grayscale datasets if needed) |
| Diversity | Multiple ethnicities, ages, genders required |
| Lighting | Mix indoor/outdoor/natural/artificial |
| Angles | Frontal + 3/4 + profile views |
| Count | 500–2,000 images per emotion (minimum) |
| Augmentation | Apply random crop, flip, brightness for more variety |

---

## 🧠 AI Model Architecture Recommendations

For a humanoid robot, consider a **multi-modal approach**:

1. **Face-only model:** ResNet-50 or EfficientNet-B3 fine-tuned on FER/AffectNet
2. **Body-language model:** Pose estimation (OpenPose) + EMOTIC-style training  
3. **Context-aware model:** Scene understanding + emotion fusion
4. **Fusion layer:** Combine all three for robust real-world performance

---

## ⚠️ Ethical Guidelines

- Always use ethically sourced, consented images
- For `Sexual_Desire` folder: use only non-explicit, subtle expression data
- Credit photographers when required (Unsplash requires attribution)
- Do not use images of children for `Romance` or `Sexual_Desire` categories
- Review each dataset's specific license before commercial use

---

## 📦 Requirements (for scripts)

```bash
pip install requests
```

That's it — both scripts only need the standard `requests` library.

---

*Generated for Humanoid Robot AI Emotion Recognition Training*
*Emotion framework: Cowen & Keltner (2017) — "Self-report captures 27 distinct categories of emotion"*

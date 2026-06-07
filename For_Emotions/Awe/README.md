# Awe

## Description
Overwhelming wonder and reverence toward something vast or powerful.

## Visual Cues for AI Training
Dropped jaw, wide eyes, frozen posture, hands over mouth.

## Recommended Search Queries
Use these on Unsplash, Pexels, Getty, or Google Images:
  - `person awestruck looking at sky`
  - `child amazed at fireworks`
  - `tourist awestruck at grand canyon`
  - `person jaw drop amazement`
  - `crowd in awe at spectacle`

## Recommended Datasets
EMOTIC, Open Images

### Dataset Links
| Dataset | URL | Notes |
|---------|-----|-------|
| FER2013 | https://www.kaggle.com/datasets/msambare/fer2013 | 35,887 grayscale facial images, 7 basic emotions |
| AffectNet | http://mohammadmahoor.com/affectnet/ | 1M+ images, 8 emotions, requires registration |
| EMOTIC | https://s3.amazonaws.com/emotic-dataset | Body + context emotion, 26 categories |
| CK+ | https://www.kaggle.com/datasets/shawon10/ckplus | Posed facial expressions, lab setting |
| JAFFE | https://zenodo.org/record/3451524 | Japanese female facial expressions |
| RAF-DB | http://www.whdeng.cn/RAF/model1.html | Real-world affective faces |
| Open Images | https://storage.googleapis.com/openimages/web/index.html | Massive diverse dataset |

## Image Guidelines for Humanoid Robot Training
- **Resolution**: Minimum 224x224px (prefer 512x512+)
- **Diversity**: Include multiple ethnicities, ages, genders
- **Lighting**: Mix of indoor, outdoor, natural, artificial
- **Angles**: Frontal, 3/4, profile views
- **Occlusion**: Some partially occluded faces for robustness
- **Count**: Aim for 500-1000 images minimum per emotion
- **Format**: JPG or PNG preferred
- **Labels**: Keep original dataset labels if available

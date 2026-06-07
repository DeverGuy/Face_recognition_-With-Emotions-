# Calmness

## Description
Peaceful, relaxed state free from stress or agitation.

## Visual Cues for AI Training
Relaxed face muscles, soft gaze, slightly upturned mouth, even breathing.

## Recommended Search Queries
Use these on Unsplash, Pexels, Getty, or Google Images:
  - `person calm serene face`
  - `woman meditating peacefully`
  - `relaxed person eyes closed`
  - `calm expression neutral face`
  - `person breathing deeply calm`

## Recommended Datasets
AffectNet, EMOTIC

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

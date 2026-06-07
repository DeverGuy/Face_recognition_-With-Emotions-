import os
import cv2
import uuid
import glob
from icrawler.builtin import BingImageCrawler

EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

QUERIES = {
    'neutral': ['calm human face portrait', 'emotionless face portrait', 'serious portrait photography'],
    'happiness': ['person smiling widely', 'happy human face portrait', 'joyful person smiling portrait'],
    'surprise': ['shocked person face', 'surprised human face portrait', 'gasping face portrait'],
    'sadness': ['crying person face', 'sad human face portrait', 'heartbroken face portrait'],
    'anger': ['furious person face', 'angry human face portrait', 'rage face portrait'],
    'disgust': ['disgusted person face', 'repulsed human face portrait', 'cringing face'],
    'fear': ['terrified person face', 'scared human face portrait', 'frightened face']
}

DATASET_DIR = "custom_dataset"
TEMP_DIR = "temp_scrape"

def extract_faces():
    print("\n[INFO] Commencing AI Face Extraction and Dataset Normalization...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(DATASET_DIR, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        raw_images = glob.glob(os.path.join(TEMP_DIR, emotion, "*.jpg"))
        success = 0
        
        for img_path in raw_images:
            img = cv2.imread(img_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                pad_w, pad_h = int(w * 0.3), int(h * 0.3)
                c_x1 = max(0, x - pad_w)
                c_y1 = max(0, y - pad_h)
                c_x2 = min(img.shape[1], x + w + pad_w)
                c_y2 = min(img.shape[0], y + h + pad_h)
                
                face_crop = img[c_y1:c_y2, c_x1:c_x2]
                face_resized = cv2.resize(face_crop, (224, 224))
                
                save_path = os.path.join(emotion_dir, f"{uuid.uuid4().hex}.jpg")
                cv2.imwrite(save_path, face_resized)
                success += 1
                break
                
        print(f"[SUCCESS] Extracted {success} high-quality faces for {emotion.upper()}.")

def main():
    print("==================================================")
    print("   BING AUTONOMOUS SCRAPER & DATASET BUILDER      ")
    print("==================================================")
    
    for emotion in EMOTIONS:
        print(f"\n[INFO] Downloading raw images for {emotion.upper()}...")
        target_dir = os.path.join(TEMP_DIR, emotion)
        os.makedirs(target_dir, exist_ok=True)
        
        crawler = BingImageCrawler(storage={'root_dir': target_dir})
        for query in QUERIES[emotion]:
            # Download 50 images per query for speed (150 total per emotion)
            crawler.crawl(keyword=query, max_num=50)
            
    extract_faces()
    print("\n[INFO] Pipeline Complete. Dataset is ready for PyTorch.")

if __name__ == "__main__":
    main()

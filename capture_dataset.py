import cv2
import os
import time

EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
DATASET_DIR = "custom_dataset"
FRAMES_PER_EMOTION = 100

def main():
    print("==================================================")
    print("       PERSONALIZED EMOTION CAPTURE TOOL          ")
    print("==================================================")
    print("This tool will record your specific micro-expressions.")
    print("Get ready to show different emotions into the camera!\n")
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
    
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(DATASET_DIR, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        print(f"\n---> PREPARE TO SHOW: {emotion.upper()}")
        for i in range(5, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
            
        print(f"[REC] Capturing {FRAMES_PER_EMOTION} frames for {emotion.upper()}...")
        
        count = 0
        while count < FRAMES_PER_EMOTION:
            ret, frame = cam.read()
            if not ret: continue
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            cv2.putText(display_frame, f"SHOW: {emotion.upper()} ({count}/{FRAMES_PER_EMOTION})", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100,100))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Crop and save
                pad_w, pad_h = int(w * 0.3), int(h * 0.3)
                c_x1 = max(0, x - pad_w)
                c_y1 = max(0, y - pad_h)
                c_x2 = min(frame.shape[1], x + w + pad_w)
                c_y2 = min(frame.shape[0], y + h + pad_h)
                
                face_crop = frame[c_y1:c_y2, c_x1:c_x2]
                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (224, 224))
                    save_path = os.path.join(emotion_dir, f"{int(time.time()*1000)}.jpg")
                    cv2.imwrite(save_path, face_resized)
                    count += 1
                break # Only one face
                
            cv2.imshow("Capture", display_frame)
            if cv2.waitKey(100) & 0xFF == 27: # ESC to quit
                print("Capture cancelled.")
                cam.release()
                cv2.destroyAllWindows()
                return
                
        print(f"[SUCCESS] Finished capturing {emotion.upper()}.")
        
    cam.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Dataset Capture Complete! You can now run `py train_custom_emotion.py`.")

if __name__ == "__main__":
    main()

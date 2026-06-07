import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_2d_106', 'recognition'])
app.prepare(ctx_id=-1, det_size=(320, 320))

img = cv2.imread("test_face.jpg")
faces = app.get(img)

if faces and hasattr(faces[0], 'landmark_2d_106'):
    lms = faces[0].landmark_2d_106
    
    # Let's see the coordinates for 33 to 42 (Left eye?)
    left_eye_indices = [35, 36, 33, 37, 39, 42, 40, 41]
    right_eye_indices = [89, 90, 87, 91, 93, 96, 94, 95]
    
    print("Left eye pts:", lms[left_eye_indices])
    print("Right eye pts:", lms[right_eye_indices])
    
    # We can calculate EAR:
    # Top points: 35, 36
    # Bottom points: 41, 40
    # Left, right points: 33, 39

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=-1, det_size=(320, 320))

img = cv2.imread("test_face.jpg")
if img is not None:
    faces = app.get(img)
    if faces and hasattr(faces[0], 'landmark_2d_106'):
        lms = faces[0].landmark_2d_106
        for i in range(106):
            pt = (int(lms[i][0]), int(lms[i][1]))
            cv2.circle(img, pt, 2, (0, 0, 255), -1)
            cv2.putText(img, str(i), (pt[0]+2, pt[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.imwrite(r"C:\Users\shris\.gemini\antigravity-ide\brain\eeff3abe-7fe8-4169-b5b9-12364d27fbb8\scratch\eye_landmarks_plot.jpg", img)
        print("Plotted to scratch directory.")

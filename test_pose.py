import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_3d_68'])
app.prepare(ctx_id=-1, det_size=(320, 320))

img = cv2.imread("test_face.jpg")
if img is not None:
    faces = app.get(img)
    if faces:
        print("Pose available:", hasattr(faces[0], 'pose'))
        if hasattr(faces[0], 'pose'):
            print("Pose:", faces[0].pose)
    else:
        print("No face detected.")
else:
    print("No test image found.")

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=-1, det_size=(320, 320))

img = cv2.imread("test_face.jpg")
if img is None:
    # create a dummy image if test_face.jpg is missing
    img = np.zeros((320, 320, 3), dtype=np.uint8)
else:
    faces = app.get(img)
    if faces and hasattr(faces[0], 'landmark_2d_106'):
        lms = faces[0].landmark_2d_106
        print("Left Eye Candidates:")
        for i in range(33, 44):
            print(f"{i}: {lms[i]}")
        print("\nRight Eye Candidates:")
        for i in range(87, 98):
            print(f"{i}: {lms[i]}")
            
        # Let's find corners by looking at x coordinates
        left_eye_pts = lms[33:44]
        min_x_idx = 33 + np.argmin(left_eye_pts[:, 0])
        max_x_idx = 33 + np.argmax(left_eye_pts[:, 0])
        print(f"\nLeft Eye Leftmost (corner): {min_x_idx}, Rightmost (corner): {max_x_idx}")
        
        # Now find top and bottom for left eye
        # Sort points between min_x and max_x
        # Actually just print the y coordinates
        min_y_idx = 33 + np.argmin(left_eye_pts[:, 1])
        max_y_idx = 33 + np.argmax(left_eye_pts[:, 1])
        print(f"Left Eye Topmost: {min_y_idx}, Bottommost: {max_y_idx}")
        
        right_eye_pts = lms[87:98]
        rmin_x_idx = 87 + np.argmin(right_eye_pts[:, 0])
        rmax_x_idx = 87 + np.argmax(right_eye_pts[:, 0])
        print(f"Right Eye Leftmost (corner): {rmin_x_idx}, Rightmost (corner): {rmax_x_idx}")
        
        rmin_y_idx = 87 + np.argmin(right_eye_pts[:, 1])
        rmax_y_idx = 87 + np.argmax(right_eye_pts[:, 1])
        print(f"Right Eye Topmost: {rmin_y_idx}, Bottommost: {rmax_y_idx}")

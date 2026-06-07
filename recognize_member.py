# pyrefly: ignore [missing-import]
import cv2
import pickle
import os
# pyrefly: ignore [missing-import]
import numpy as np

def load_camera_source(config_path="camera_config.txt"):
    # Default source
    source = 0
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                content = f.read().strip()
                if content.isdigit():
                    source = int(content)
                else:
                    source = content
        except Exception:
            pass
            
    print("\n==========================================")
    print("        BIOMETRIC CAMERA SELECTOR         ")
    print("==========================================")
    
    if os.path.exists(config_path):
        print(f"Current Configured Source: {source}")
        choice = input("Press [ENTER] to use this source, or type 'n' to reconfigure: ").strip().lower()
        if choice != 'n':
            return source
            
    print("\nSelect Camera Option:")
    print(" [1] Laptop Built-in Camera")
    print(" [2] Wireless Phone IP Camera (Wi-Fi)")
    print(" [3] Virtual Camera (DroidCam, Iriun, etc.)")
    
    opt = input("Option (1-3) [default: 1]: ").strip()
    if opt == "2":
        url = input("Enter Phone IP URL (e.g., http://192.168.1.50:8080/video): ").strip()
        if not url.startswith("http"):
            url = "http://" + url
        source = url
    elif opt == "3":
        idx = input("Enter camera index (1, 2, etc.) [default: 1]: ").strip()
        source = int(idx) if idx.isdigit() else 1
    else:
        source = 0
        
    try:
        with open(config_path, "w") as f:
            f.write(str(source))
        print(f"Configuration saved to {config_path}\n")
    except Exception as e:
        print(f"Warning: Could not save configuration: {e}")
        
    return source

CAMERA_SOURCE = load_camera_source()

# pyrefly: ignore [missing-import]
import onnxruntime as ort
import threading
import time
import random

# Monkey-patch ONNX Runtime to optimize performance on Windows (limit CPU threads to prevent lag)
original_init = ort.InferenceSession.__init__
def patched_init(self, model_path, sess_options=None, providers=None, provider_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    # Limit CPU threads to prevent 100% core usage lag
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 2
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Try GPU execution providers first, then CPU
    providers = ['CUDAExecutionProvider', 'DirectMLExecutionProvider', 'CPUExecutionProvider']
    original_init(self, model_path, sess_options, providers, provider_options, **kwargs)
ort.InferenceSession.__init__ = patched_init

# pyrefly: ignore [missing-import]
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis optimized for speed (detection, landmark extraction, pose, and recognition)
# We set det_thresh=0.65 to eliminate false detections on background patterns
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_2d_106', 'landmark_3d_68', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.65)

# Try to load YOLO for Object-Aware Attention Tracking
try:
    # pyrefly: ignore [missing-import]
    from ultralytics import YOLO
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    yolo_model = YOLO("yolov8n.pt", verbose=False)
    has_yolo = True
except ImportError:
    has_yolo = False

# Load database and auto-normalize embeddings
if os.path.exists("embeddings.pkl") and os.path.getsize("embeddings.pkl") > 0:
    with open("embeddings.pkl", "rb") as f:
        database = pickle.load(f)
    modified = False
    for name, stored_val in database.items():
        if isinstance(stored_val, list):
            normalized_list = []
            for emb in stored_val:
                norm = np.linalg.norm(emb)
                if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
                    normalized_list.append(emb / norm)
                    modified = True
                else:
                    normalized_list.append(emb)
            database[name] = normalized_list
        elif isinstance(stored_val, np.ndarray) and stored_val.ndim == 2:
            normalized_rows = []
            for emb in stored_val:
                norm = np.linalg.norm(emb)
                if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
                    normalized_rows.append(emb / norm)
                    modified = True
                else:
                    normalized_rows.append(emb)
            database[name] = np.array(normalized_rows)
        else:
            # Single 1D array (legacy format)
            norm = np.linalg.norm(stored_val)
            if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
                database[name] = stored_val / norm
                modified = True
    if modified:
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(database, f)
        print("Auto-normalized existing database embeddings.")
else:
    database = {}

# Thread-safe variables
frame_to_process = None
processed_results = []
processed_objects = []
results_lock = threading.Lock()
frame_lock = threading.Lock()
running = True

# 1. Threaded camera reader to eliminate OpenCV frame buffering lag
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"\n[ERROR] Could not open camera source: {src}")
            print("If you are using a Wi-Fi IP Camera:")
            print(" 1. Ensure your phone and laptop are connected to the exact same Wi-Fi network.")
            print(" 2. Ensure the IP address and port match what is shown on your phone's app.")
            print(" 3. Try opening the URL in your web browser first to verify the stream is working.\n")
            os._exit(1)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            try:
                ret = self.cap.grab()
                if ret:
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None:
                        self.ret = ret
                        self.frame = frame
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

# 2. Worker thread for running face inference asynchronously
def worker():
    global frame_to_process, processed_results, running
    while running:
        frame = None
        with frame_lock:
            if frame_to_process is not None:
                frame = frame_to_process.copy()
                frame_to_process = None  # Consume frame
        
        if frame is not None:
            # Scale frame down for faster model inference (640px width is perfect for crowds)
            h, w = frame.shape[:2]
            target_width = 640
            scale = target_width / w
            if scale < 1.0:
                target_height = int(h * scale)
                inference_frame = cv2.resize(frame, (target_width, target_height))
            else:
                scale = 1.0
                inference_frame = frame
            
            # 1. Detect objects with YOLOv8 if available
            detected_objects = []
            if has_yolo:
                try:
                    # Lowered confidence threshold to 0.15 to catch phones held at weird angles
                    yolo_results = yolo_model(inference_frame, classes=[67, 73], conf=0.15, verbose=False) # 67=cell phone, 73=laptop
                    if len(yolo_results) > 0:
                        for box in yolo_results[0].boxes:
                            cls_id = int(box.cls[0])
                            obj_name = "phone" if cls_id == 67 else "laptop"
                            ox1, oy1, ox2, oy2 = box.xyxy[0].cpu().numpy()
                            detected_objects.append({
                                "name": obj_name,
                                "bbox": [int(ox1 / scale), int(oy1 / scale), int(ox2 / scale), int(oy2 / scale)]
                            })
                except Exception:
                    pass
            
            # 2. Detect Faces
            faces = app.get(inference_frame)
            results = []
            for face in faces:
                # Filter out garbage false positives (like chairs), but keep it low enough
                # to catch actual people sitting far in the background.
                if face.det_score < 0.45:
                    continue
                
                emb = face.embedding
                emb_norm = np.linalg.norm(emb)
                
                # Calculate Eye Aspect Ratio (EAR) for liveness detection (Blink)
                ear = 1.0
                mar = 0.0
                if hasattr(face, 'landmark_2d_106'):
                    lms = face.landmark_2d_106
                    # Left eye indices
                    pt35, pt39 = lms[35], lms[39] # corners
                    pt41, pt36 = lms[41], lms[36] # left vertical
                    pt42, pt37 = lms[42], lms[37] # right vertical
                    v_left1 = np.linalg.norm(pt41 - pt36)
                    v_left2 = np.linalg.norm(pt42 - pt37)
                    h_left = np.linalg.norm(pt35 - pt39)
                    ear_left = (v_left1 + v_left2) / (2.0 * h_left + 1e-6)

                    # Right eye indices
                    pt89, pt93 = lms[89], lms[93] # corners
                    pt95, pt90 = lms[95], lms[90] # left vertical
                    pt96, pt91 = lms[96], lms[91] # right vertical
                    v_right1 = np.linalg.norm(pt95 - pt90)
                    v_right2 = np.linalg.norm(pt96 - pt91)
                    h_right = np.linalg.norm(pt89 - pt93)
                    ear_right = (v_right1 + v_right2) / (2.0 * h_right + 1e-6)

                    ear = (ear_left + ear_right) / 2.0
                    
                    # Calculate Mouth Aspect Ratio (MAR) for smile/open mouth
                    pt52, pt61 = lms[52], lms[61]
                    pt71, pt66 = lms[71], lms[66]
                    h_mouth = np.linalg.norm(pt52 - pt61)
                    v_mouth = np.linalg.norm(pt71 - pt66)
                    mar = v_mouth / (h_mouth + 1e-6)
                
                emb = emb / emb_norm if emb_norm > 0 else emb
                
                best_name = "Unknown"
                best_distance = 999999
                
                for name, stored_val in database.items():
                    # Handle multi-sample list/2D array and legacy 1D array formats
                    stored_embs = np.atleast_2d(stored_val)
                    distances = np.linalg.norm(stored_embs - emb, axis=1)
                    min_dist = np.min(distances)
                    if min_dist < best_distance:
                        best_distance = min_dist
                        best_name = name
                
                # Relaxed L2 threshold (1.10) so it successfully identifies registered members
                # even if they are far in the back, in side-profile, or covering their mouths.
                if best_distance > 1.10:
                    best_name = "Unknown"
                
                # Scale face bounding box coordinates back to full resolution
                x1, y1, x2, y2 = face.bbox
                x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
                
                # Get head pose (pitch, yaw, roll)
                pose = [0, 0, 0]
                if hasattr(face, 'pose') and face.pose is not None:
                    pose = face.pose
                
                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "det_score": face.det_score,
                    "embedding": emb,
                    "name": best_name,
                    "distance": best_distance,
                    "ear": ear,
                    "mar": mar,
                    "pose": pose,
                    "kps": face.kps
                })
            
            with results_lock:
                processed_results = results
                processed_objects = detected_objects
        else:
            time.sleep(0.005)

# Start inference worker thread
t = threading.Thread(target=worker, daemon=True)
t.start()

# Start camera capture thread
cam = CameraStream(CAMERA_SOURCE)

# Cyberpunk HUD UI box drawing helper
def draw_cyber_box(frame, bbox, name, distance=None, liveness_verified=False, challenge="blink", locked=False, ear=1.0, mar=0.0, pose=[0, 0, 0], kps=None, objects=[]):
    x1, y1, x2, y2 = bbox
    w_box = x2 - x1
    h_box = y2 - y1
    
    # BGR colors based on liveness and recognition status
    if name != "Unknown":
        if liveness_verified:
            color = (80, 255, 100)   # Neon/Emerald Green (BGR)
            status_tag = "SECURED"
        else:
            color = (0, 200, 255)    # Cyber Yellow for pending liveness
            status_tag = "LIVENESS UNVERIFIED"
    else:
        color = (0, 75, 255)     # Cyber Neon Orange/Red (BGR)
        status_tag = "UNAUTHORIZED"
        
    # 1. Glowing outer box outlines
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 1, lineType=cv2.LINE_AA) # Black drop shadow
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA) # Main colored boundary
    
    # 2. Sleek thick corner brackets
    corner_len = min(22, int(w_box * 0.18), int(h_box * 0.18))
    thickness = 3
    
    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness, lineType=cv2.LINE_AA)
    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness, lineType=cv2.LINE_AA)
    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness, lineType=cv2.LINE_AA)
    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness, lineType=cv2.LINE_AA)
    
    # 3. Sweeping neon scanline
    if not locked:
        scan_period = 2.0  # Seconds
        t_cycle = (time.time() % scan_period) / scan_period
        pos = t_cycle * 2 if t_cycle < 0.5 else (1.0 - t_cycle) * 2
        scan_y = int(y1 + pos * h_box)
        
        # Draw scanline with side ticks
        cv2.line(frame, (x1, scan_y), (x2, scan_y), color, 1, lineType=cv2.LINE_AA)
        cv2.line(frame, (x1, scan_y - 2), (x1 + 5, scan_y - 2), color, 1, lineType=cv2.LINE_AA)
        cv2.line(frame, (x2 - 5, scan_y - 2), (x2, scan_y - 2), color, 1, lineType=cv2.LINE_AA)

    # 6. Draw Behavior / Attention Tracker using 3D Head Pose & Objects
    pitch, yaw, roll = pose
    
    # Calculate mathematically rock-solid YAW using 2D Facial Landmarks
    # 1. Nose-Offset Ratio: As the head turns, the nose moves horizontally far away from the midpoint between the eyes.
    is_looking_away = False
    if kps is not None and len(kps) >= 3:
        # Euclidean distance between left and right eye
        eye_dist = ((kps[0][0] - kps[1][0])**2 + (kps[0][1] - kps[1][1])**2)**0.5
        eye_mid_x = (kps[0][0] + kps[1][0]) / 2.0
        nose_offset = abs(kps[2][0] - eye_mid_x)
        
        # If the nose offset from the center of the eyes exceeds 65% of the eye distance, 
        # the head is severely turned to the side.
        if eye_dist > 0 and (nose_offset / eye_dist) > 0.65:
            is_looking_away = True
    
    # Check for YOLO Object Overlaps to override head pose
    using_phone = False
    using_laptop = False
    for obj in objects:
        ox1, oy1, ox2, oy2 = obj["bbox"]
        obj_cx = (ox1 + ox2) / 2
        
        if obj["name"] == "phone":
            # If phone is near the face horizontally and vertically
            if x1 - 100 < obj_cx < x2 + 100 and oy1 < y2 + 350:
                using_phone = True
        elif obj["name"] == "laptop":
            # If laptop is roughly underneath the face
            if x1 - 150 < obj_cx < x2 + 150 and oy1 > y1:
                using_laptop = True
    
    # Categorize behavior based on angles (Note: pitch > 0 is looking down, pitch < 0 is looking up)
    behavior_prefix = f"{name.upper()}'S BEHAVIOR" if name != "Unknown" else "BEHAVIOR"
    
    if is_looking_away:
        behavior = f"{behavior_prefix}: DISTRACTED (LOOKING AWAY)"
        behavior_color = (0, 100, 255) # Orange
    elif pitch < -25:
        behavior = f"{behavior_prefix}: LOOKING DOWN AT THEIR BEAN"
        behavior_color = (255, 150, 0) # Light blue (BGR)
    elif pitch > 25:
        behavior = f"{behavior_prefix}: DISTRACTED (LOOKING UP)"
        behavior_color = (0, 100, 255) # Orange
    else:
        behavior = f"{behavior_prefix}: DIRECT ATTENTION"
        behavior_color = (255, 255, 0) # Cyan
        
    # OVERRIDE with Object Logic!
    if using_phone:
        behavior = f"{behavior_prefix}: ON PHONE (DISTRACTED)"
        behavior_color = (0, 0, 255) # Red
    elif using_laptop:
        behavior = f"{behavior_prefix}: WORKING ON LAPTOP"
        behavior_color = (0, 255, 0) # Green

    # Add black background for behavior text to make it perfectly readable
    (tw, th), _ = cv2.getTextSize(behavior, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(frame, (x1, y2 + 5), (x1 + tw + 4, y2 + 5 + th + 4), (0, 0, 0), -1)
    cv2.putText(frame, behavior, (x1 + 2, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, behavior_color, 1, lineType=cv2.LINE_AA)
    
    # 4. Futuristic Text Badge
    if name != "Unknown":
        if liveness_verified:
            if distance is not None:
                # Distance confidence mapping for ArcFace embeddings
                # L2 distance of 0.7-0.8 represents a highly confident match
                if distance <= 0.6:
                    match_pct = int(98 + (0.6 - distance) * 3) # Caps near 99%
                elif distance <= 0.8:
                    match_pct = int(90 + (0.8 - distance) / 0.2 * 8)
                elif distance <= 0.9:
                    match_pct = int(80 + (0.9 - distance) / 0.1 * 10)
                else:
                    match_pct = 0
                label = f"{status_tag} // {name.upper()} // {match_pct}%"
            else:
                label = f"{status_tag} // {name.upper()}"
        else:
            # Show instructions for liveness challenge and live telemetry
            label = f"{status_tag} // BLINK EYES TO VERIFY [EAR: {ear:.2f}]"
    else:
        label = f"{status_tag} // EXCLUDE"
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    text_thickness = 1
    
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    
    frame_h, frame_w = frame.shape[:2]
    badge_x1 = x1
    badge_x2 = x1 + text_w + 14
    
    # Shift badge left if it goes off the right edge
    if badge_x2 > frame_w:
        badge_x1 -= (badge_x2 - frame_w)
        badge_x2 = frame_w
        if badge_x1 < 0:
            badge_x1 = 0
            badge_x2 = text_w + 14

    badge_y1 = max(0, y1 - text_h - 10)
    badge_y2 = y1
    # If clipped at the top, move badge below the bounding box
    if badge_y2 - badge_y1 < text_h + 5:
        badge_y1 = y2
        badge_y2 = y2 + text_h + 10
    
    # Draw dark shadow behind badge first
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), -1)
    # Draw colored border around badge
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, lineType=cv2.LINE_AA)
    # Render text inside badge
    cv2.putText(frame, label, (badge_x1 + 7, badge_y2 - 6), font, font_scale, color, text_thickness, lineType=cv2.LINE_AA)
    
    # 5. Central crosshair target dot
    cx, cy = x1 + w_box // 2, y1 + h_box // 2
    cv2.circle(frame, (cx, cy), 2, color, -1)

# Real-time face tracking variables
tracked_faces = {}
next_face_id = 0
LERP_FACTOR = 0.35    # Smoother display box interpolation

last_results = None

# FPS calculation variables
fps_start_time = time.time()
fps_counter = 0
fps = 30.0

cv2.namedWindow("Club Robot", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty("Club Robot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def show_fullscreen(window_name, img):
    dh, dw = img.shape[:2]
    target_w = int(dh * 16 / 9)
    if target_w > dw:
        canvas = np.zeros((dh, target_w, 3), dtype=np.uint8)
        x_off = (target_w - dw) // 2
        canvas[:, x_off:x_off+dw] = img
        cv2.imshow(window_name, canvas)
    else:
        cv2.imshow(window_name, img)

# Video rotation state
rotation_state = 1  # 0: None, 1: 90 CW, 2: 180, 3: 90 CCW

while True:
    ret, raw_frame = cam.read()
    if not ret or raw_frame is None:
        continue
    
    # Only mirror if it's a local built-in webcam
    if isinstance(CAMERA_SOURCE, int):
        frame = cv2.flip(raw_frame, 1)
    else:
        frame = raw_frame.copy()
        if rotation_state == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_state == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_state == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Prevent UI from becoming microscopic on high-res phone cameras
    h, w = frame.shape[:2]
    max_h = 720
    if h > max_h:
        scale_d = max_h / h
        frame = cv2.resize(frame, (int(w * scale_d), max_h))
        
    h, w = frame.shape[:2]
    
    # Pass frame to worker if it is ready to process
    with frame_lock:
        if frame_to_process is None:
            frame_to_process = frame.copy()
            
    # Check if there are new processed results from the worker thread
    new_results_available = False
    current_results = []
    current_objects = []
    with results_lock:
        if processed_results is not last_results:
            current_results = list(processed_results)
            current_objects = list(processed_objects)
            last_results = processed_results
            new_results_available = True

    current_time = time.time()
    
    # 1. Update Tracking with AI Results (when available)
    if new_results_available:
        matched_tracked = set()
        matched_results = set()
        
        potential_matches = []
        for res_idx, res in enumerate(current_results):
            rx1, ry1, rx2, ry2 = res["bbox"]
            rcx, rcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            face_size = max(rx2 - rx1, ry2 - ry1)
            
            for fid, f in tracked_faces.items():
                fx1, fy1, fx2, fy2 = f["tracker_bbox"]
                fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                dist = np.sqrt((rcx - fcx)**2 + (ry1 - fcy)**2)
                
                # Dynamic threshold: allow fast movement
                if dist < max(350, face_size * 2.5):
                    potential_matches.append((dist, res_idx, fid))
        
        # Sort matches closest first
        potential_matches.sort(key=lambda x: x[0])
        
        # Bipartite matching association
        for dist, res_idx, fid in potential_matches:
            if res_idx not in matched_results and fid not in matched_tracked:
                res = current_results[res_idx]
                tracked_faces[fid]["tracker_bbox"] = list(res["bbox"])
                tracked_faces[fid]["name"] = res["name"]
                tracked_faces[fid]["distance"] = res["distance"]
                tracked_faces[fid]["last_seen"] = current_time
                
                # Update liveness variables only if recognized
                ear = res["ear"]
                mar = res["mar"]
                tracked_faces[fid]["ear"] = ear
                tracked_faces[fid]["mar"] = mar
                
                if res["name"] != "Unknown":
                    # Lock the tracking if it's a member
                    tracked_faces[fid]["locked"] = True
                    
                    if "min_ear" not in tracked_faces[fid]:
                        tracked_faces[fid]["min_ear"] = ear
                        tracked_faces[fid]["max_ear"] = ear
                        tracked_faces[fid]["min_mar"] = mar
                        tracked_faces[fid]["max_mar"] = mar
                        
                    tracked_faces[fid]["name"] = res["name"]
                    tracked_faces[fid]["distance"] = res["distance"]
                    tracked_faces[fid]["ear"] = res["ear"]
                    tracked_faces[fid]["mar"] = res["mar"]
                    tracked_faces[fid]["pose"] = res["pose"]
                    
                    tracked_faces[fid]["min_ear"] = min(tracked_faces[fid]["min_ear"], ear)
                    tracked_faces[fid]["max_ear"] = max(tracked_faces[fid]["max_ear"], ear)
                    tracked_faces[fid]["min_mar"] = min(tracked_faces[fid]["min_mar"], mar)
                    tracked_faces[fid]["max_mar"] = max(tracked_faces[fid]["max_mar"], mar)
                    
                    # Verify liveness based on the assigned challenge
                    if not tracked_faces[fid].get("liveness_verified", False):
                        # Hyper-sensitive but secure blink threshold (0.05 diff)
                        # Adapts to soft blinks but still requires an actual change to prevent static photos.
                        ear_diff = tracked_faces[fid]["max_ear"] - tracked_faces[fid]["min_ear"]
                        if ear_diff > 0.05:
                            tracked_faces[fid]["liveness_verified"] = True
                else:
                    tracked_faces[fid]["locked"] = False

                # Crop a new template to track in intermediate frames
                x1, y1, x2, y2 = res["bbox"]
                x1_c = max(0, x1)
                y1_c = max(0, y1)
                x2_c = min(w, x2)
                y2_c = min(h, y2)
                if x2_c > x1_c and y2_c > y1_c:
                    tracked_faces[fid]["template"] = frame[y1_c:y2_c, x1_c:x2_c].copy()
                    
                matched_results.add(res_idx)
                matched_tracked.add(fid)
                
        # Create new tracked faces for unmatched AI detections
        for res_idx, res in enumerate(current_results):
            if res_idx not in matched_results:
                x1, y1, x2, y2 = res["bbox"]
                x1_c = max(0, x1)
                y1_c = max(0, y1)
                x2_c = min(w, x2)
                y2_c = min(h, y2)
                
                new_face = {
                    "bbox": list(res["bbox"]),
                    "tracker_bbox": list(res["bbox"]),
                    "name": res["name"],
                    "distance": res["distance"],
                    "last_seen": current_time,
                    "ear": res["ear"],
                    "mar": res["mar"],
                    "pose": res["pose"],
                    "kps": res.get("kps", None),
                    "liveness_verified": False,
                    "challenge": "blink",
                    "locked": False
                }
                
                if x2_c > x1_c and y2_c > y1_c:
                    new_face["template"] = frame[y1_c:y2_c, x1_c:x2_c].copy()
                    
                tracked_faces[next_face_id] = new_face
                next_face_id += 1

    # 2. Intermediate Frame Template Tracking (Main Thread - 30 FPS)
    # If the AI is busy, we track the face position using template matching
    for fid, f in tracked_faces.items():
        if f.get("locked", False):
            # Target is locked; pause intermediate tracking to stabilize the box fully.
            continue
            
        if "template" in f and f["template"] is not None:
            tx1, ty1, tx2, ty2 = f["tracker_bbox"]
            tw = tx2 - tx1
            th = ty2 - ty1
            
            # Pad the search window to allow for rapid face movement
            pad_w = tw // 2
            pad_h = th // 2
            
            sx1 = max(0, tx1 - pad_w)
            sy1 = max(0, ty1 - pad_h)
            sx2 = min(w, tx2 + pad_w)
            sy2 = min(h, ty2 + pad_h)
            
            # Verify valid dimensions for template matching
            if (sy2 - sy1) > th and (sx2 - sx1) > tw and th > 15 and tw > 15:
                search_region = frame[sy1:sy2, sx1:sx2]
                template_img = f["template"]
                
                try:
                    res = cv2.matchTemplate(search_region, template_img, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    
                    if max_val > 0.65: # Stricter match score validation to prevent background drift
                        dx, dy = max_loc
                        new_x1 = sx1 + dx
                        new_y1 = sy1 + dy
                        new_x2 = new_x1 + tw
                        new_y2 = new_y1 + th
                        
                        f["tracker_bbox"] = [new_x1, new_y1, new_x2, new_y2]
                        
                        # Grab a fresh template from this new position to handle rotation/tilt
                        nx1_c = max(0, new_x1)
                        ny1_c = max(0, new_y1)
                        nx2_c = min(w, new_x2)
                        ny2_c = min(h, new_y2)
                        if nx2_c > nx1_c and ny2_c > ny1_c:
                            f["template"] = frame[ny1_c:ny2_c, nx1_c:nx2_c].copy()
                except Exception:
                    pass

    # 3. Clean up stale tracked faces (timeout if not validated by AI for > 1.5s)
    to_delete = [fid for fid, f in tracked_faces.items() if current_time - f["last_seen"] > 1.5]
    for fid in to_delete:
        del tracked_faces[fid]

    # 4. Interpolate display bboxes (Lerp) to smoothly draw transitions
    for fid, f in tracked_faces.items():
        curr = f["bbox"]
        target = f["tracker_bbox"]
        # Use stronger smoothing (lower LERP) if the face is locked
        lerp = 0.15 if f.get("locked", False) else LERP_FACTOR
        for i in range(4):
            curr[i] = int(curr[i] + lerp * (target[i] - curr[i]))
        f["bbox"] = curr

    # 5. Draw Cyber HUD Elements
    
    # Glassmorphic telemetry panel (top-left)
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (220, 115), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    
    # Render HUD texts
    cv2.putText(frame, "BIOMETRIC MONITOR", (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, "SYSTEM: ACTIVE", (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"VIDEO FPS: {fps:.1f}", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"TARGETS DETECTED: {len(tracked_faces)}", (25, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(
        frame, 
        "TARGET LOCK: SECURED" if len(tracked_faces) > 0 else "TARGET LOCK: STANDBY", 
        (25, 100), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.38, 
        (80, 255, 100) if len(tracked_faces) > 0 else (120, 120, 120), 
        1, 
        lineType=cv2.LINE_AA
    )

    # Draw tracking boxes for active subjects
    for fid, f in tracked_faces.items():
        draw_cyber_box(
            frame, 
            f["bbox"], 
            f["name"], 
            f.get("distance"),
            f.get("liveness_verified", False),
            f.get("challenge", "blink"),
            f.get("locked", False),
            f.get("ear", 1.0),
            f.get("mar", 0.0),
            f.get("pose", [0, 0, 0]),
            f.get("kps", None),
            current_objects
        )
        
    # Draw detected objects as thin gray boxes for visualization
    for obj in current_objects:
        ox1, oy1, ox2, oy2 = obj["bbox"]
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (100, 100, 100), 1)
        cv2.putText(frame, obj["name"].upper(), (ox1, max(15, oy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    # Draw rotation hint
    if not isinstance(CAMERA_SOURCE, int):
        cv2.putText(frame, "PRESS 'R' TO ROTATE CAMERA", (25, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 150, 255), 1, lineType=cv2.LINE_AA)
    
    show_fullscreen("Club Robot", frame)
    
    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_start_time > 1.0:
        fps = fps_counter / (time.time() - fps_start_time)
        fps_counter = 0
        fps_start_time = time.time()
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        rotation_state = (rotation_state + 1) % 4

running = False
cam.release()
cv2.destroyAllWindows()
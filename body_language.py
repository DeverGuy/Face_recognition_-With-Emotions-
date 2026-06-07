import cv2
import mediapipe as mp
import numpy as np

class BodyLanguageAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def analyze(self, frame):
        """
        Analyzes full body pose to extract body language emotions.
        Returns: emotion_override (str or None), processed_frame
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        action_emotion = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Helper to get coordinates
            def get_coord(idx):
                lm = landmarks[idx]
                return (lm.x, lm.y, lm.z, lm.visibility)
                
            # Keypoints
            nose = get_coord(self.mp_pose.PoseLandmark.NOSE.value)
            left_eye = get_coord(self.mp_pose.PoseLandmark.LEFT_EYE.value)
            right_eye = get_coord(self.mp_pose.PoseLandmark.RIGHT_EYE.value)
            left_wrist = get_coord(self.mp_pose.PoseLandmark.LEFT_WRIST.value)
            right_wrist = get_coord(self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
            left_elbow = get_coord(self.mp_pose.PoseLandmark.LEFT_ELBOW.value)
            right_elbow = get_coord(self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            left_shoulder = get_coord(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            right_shoulder = get_coord(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            
            # More Keypoints
            left_hip = get_coord(self.mp_pose.PoseLandmark.LEFT_HIP.value)
            right_hip = get_coord(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
            
            # Visibilities
            hands_visible = left_wrist[3] > 0.5 and right_wrist[3] > 0.5
            one_hand_visible = left_wrist[3] > 0.5 or right_wrist[3] > 0.5
            arms_visible = left_elbow[3] > 0.5 and right_elbow[3] > 0.5
            
            if hands_visible:
                # 1. Hands covering face / Head in hands -> Frustrated / Sadness
                # Both wrists are very close to the nose in Y, and inside the shoulders in X
                hands_near_face_y = abs(left_wrist[1] - nose[1]) < 0.15 and abs(right_wrist[1] - nose[1]) < 0.15
                if hands_near_face_y:
                    action_emotion = "frustrated"
                    
                # 4. Hands thrown up in the air -> Surprised / Shocked
                # Wrists are significantly higher (lower Y) than the shoulders/nose
                hands_up = left_wrist[1] < left_shoulder[1] - 0.2 and right_wrist[1] < right_shoulder[1] - 0.2
                if hands_up:
                    action_emotion = "shocked"
                    
                # 6. Thumbs up or Clapping -> Euphoric
                # Clapping: Wrists are very close to each other in front of the body
                clapping = abs(left_wrist[0] - right_wrist[0]) < 0.05 and abs(left_wrist[1] - right_wrist[1]) < 0.05
                if clapping:
                    action_emotion = "euphoric"
                    
                # 7. Hands on hips (Power pose) -> Confident/Euphoric
                hands_on_hips = (abs(left_wrist[0] - left_hip[0]) < 0.15 and abs(left_wrist[1] - left_hip[1]) < 0.15) and \
                                (abs(right_wrist[0] - right_hip[0]) < 0.15 and abs(right_wrist[1] - right_hip[1]) < 0.15)
                if hands_on_hips:
                    action_emotion = "euphoric"

            if one_hand_visible and not action_emotion:
                # 8. Waving hand -> Happiness
                waving_left = left_wrist[1] < left_shoulder[1] and left_wrist[0] > left_shoulder[0] + 0.15
                waving_right = right_wrist[1] < right_shoulder[1] and right_wrist[0] < right_shoulder[0] - 0.15
                if waving_left or waving_right:
                    action_emotion = "happiness"

                # 9. Rubbing back of neck -> Confused
                # Wrist is near shoulder/neck level but Z-depth is behind the nose
                left_on_neck = abs(left_wrist[0] - nose[0]) < 0.15 and left_wrist[1] < left_shoulder[1] + 0.1 and left_wrist[2] > nose[2]
                right_on_neck = abs(right_wrist[0] - nose[0]) < 0.15 and right_wrist[1] < right_shoulder[1] + 0.1 and right_wrist[2] > nose[2]
                if left_on_neck or right_on_neck:
                    action_emotion = "confused"
                    
                # 10. Thinker pose (hand on chin) -> Suspicious / Confused
                left_on_chin = abs(left_wrist[0] - nose[0]) < 0.1 and abs(left_wrist[1] - (nose[1] + 0.1)) < 0.08
                right_on_chin = abs(right_wrist[0] - nose[0]) < 0.1 and abs(right_wrist[1] - (nose[1] + 0.1)) < 0.08
                if left_on_chin or right_on_chin:
                    action_emotion = "suspicious"
            
            if arms_visible and not action_emotion:
                # 2. Arms crossed tightly over chest -> Defensive / Angry
                # Wrists are crossed over the opposite side of the body
                crossed_left = left_wrist[0] > right_shoulder[0]
                crossed_right = right_wrist[0] < left_shoulder[0]
                wrists_near_chest = abs(left_wrist[1] - left_shoulder[1]) < 0.2 and abs(right_wrist[1] - right_shoulder[1]) < 0.2
                if crossed_left and crossed_right and wrists_near_chest:
                    action_emotion = "anger"
                    
            if not action_emotion:
                # 3. Slouching heavily -> Bored / Exhausted
                # Shoulders are significantly lower compared to a normal straight posture (harder to detect purely from Y, 
                # but we can look at shoulder width and neck hunching if we use Z depth or relative distance)
                # Let's use a proxy: if nose is very close to shoulders in Y (hunching)
                hunching = abs(nose[1] - ((left_shoulder[1] + right_shoulder[1])/2)) < 0.1
                if hunching:
                    action_emotion = "exhausted"

            # Draw the skeleton for HUD feedback
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
            )
            
        return action_emotion, frame

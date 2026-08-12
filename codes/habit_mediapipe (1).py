"""
create new module for mediapipe as the code can be too long
will help in making the code cleaner and easy to fine tune
plus, finding errors specifically in mediapipe

Habit Detection Algorithm
Detects: head scratching, nose scratching, head tilting, hand-to-face movements
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum

class HabitBehavior(Enum):
    """Habit-related behaviors"""
    #HEAD_SCRATCHING = "Head Scratching"
    NOSE_SCRATCHING = "Nose Scratching"
    #HEAD_TILTING = "Head Tilting"
    #FACE_TOUCHING = "Face Touching"
    #EAR_TOUCHING = "Ear Touching"
    NECK_RUBBING = "Neck Rubbing"
    #HAND_WRINGING = "Hand Wringing"
    FIDGETING = "Fidgeting"
    NONE = "No Behavior Detected"

@dataclass
class HabitDetectionResult:
    """Result of habit behavior detection"""
    behavior: HabitBehavior
    confidence: float  # 0.0 to 1.0
    duration_frames: int
    is_active: bool


class HabitDetector:
    """
    Detects habit-related habits using pose and hand landmarks
    
    Usage:
        detector = HabitDetector(max_people=4)
        
        # Each frame:
        results = detector.detect(pose_landmarks_list, hand_landmarks_list)

        for result in results:
            print(f"Person {result.person_id}: {result.behavior.value}")        
    """
    
    def __init__(self,  history_size=15, confidence_threshold=0.5):
        """
        Args:
            history_size: Number of frames to track for temporal patterns
            confidence_threshold: Minimum confidence to classify behavior
        """
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        
        # Track people across frames
        self.behavior_history = deque(maxlen=history_size)
        self.frame_count = 0
        self.current_behavior = HabitBehavior.NONE
        self.behavior_duration = 0
        
        # Head position history for tilting detection
        self.head_pos_history = deque(maxlen=history_size)


    def detect(self, pose_landmarks, hand_landmarks_list):
        """
        Detect habit behaviors from landmarks
        
        Args:
            pose_landmarks: List of pose landmarks (33 points)
            hand_landmarks_list: List of hand landmark lists (21 points each)
            
        Returns:
            HabitDetectionResult
        """
        self.frame_count += 1

        # Extract key landmarks
        head_landmarks = self._extract_head_landmarks(pose_landmarks)
        hand_face_distance = self._calculate_hand_face_distance(
            pose_landmarks, hand_landmarks_list
        )
        hand_movement = self._calculate_hand_movement(hand_landmarks_list)
        head_tilt = self._calculate_head_tilt(pose_landmarks)
        
        # Detect behaviors
        behavior_scores = {
            #HabitBehavior.HEAD_SCRATCHING: self._detect_head_scratching(
            #   hand_face_distance, hand_movement
            #),
            HabitBehavior.NOSE_SCRATCHING: self._detect_nose_scratching(
                pose_landmarks, hand_landmarks_list
            ),
            #HabitBehavior.HEAD_TILTING: self._detect_head_tilting(head_tilt),
            #HabitBehavior.FACE_TOUCHING: self._detect_face_touching(
            #    hand_face_distance
            #),
            #HabitBehavior.EAR_TOUCHING: self._detect_ear_touching(
            #    pose_landmarks, hand_landmarks_list
            #),
            HabitBehavior.NECK_RUBBING: self._detect_neck_rubbing(
                pose_landmarks, hand_landmarks_list
            ),
            #HabitBehavior.HAND_WRINGING: self._detect_hand_wringing(
            #    hand_landmarks_list
            #),
            HabitBehavior.FIDGETING: self._detect_fidgeting(hand_movement),
        }
        
        # Get highest confidence behavior
        best_behavior = max(behavior_scores, key=behavior_scores.get)
        best_confidence = behavior_scores[best_behavior]
        
        # Update behavior tracking
        if best_confidence >= self.confidence_threshold:
            if best_behavior == self.current_behavior:
                self.behavior_duration += 1
            else:
                self.current_behavior = best_behavior
                self.behavior_duration = 1
        else:
            self.current_behavior = HabitBehavior.NONE
            self.behavior_duration = 0
        
        self.behavior_history.append((best_behavior, best_confidence))
        
        return HabitDetectionResult(
            behavior=self.current_behavior,
            confidence=best_confidence,
            duration_frames=self.behavior_duration,
            is_active=best_confidence >= self.confidence_threshold
        )
     
    
    # ==================== HELPER METHODS ====================
    

    def _extract_head_landmarks(self, pose_landmarks):
        #Extract head-related landmarks
        return {
            'nose': pose_landmarks[0],
            'left_eye': pose_landmarks[2],
            'right_eye': pose_landmarks[5],
            'left_ear': pose_landmarks[3],
            'right_ear': pose_landmarks[6],
        }


    def _calculate_hand_face_distance(self, pose_landmarks, hand_landmarks_list):
        """Calculate minimum distance from hands to face region"""
        if not hand_landmarks_list:
            return float('inf')
        
        nose = pose_landmarks[0]
        left_eye = pose_landmarks[2]
        right_eye = pose_landmarks[5]
        
        face_points = [
            (nose.x, nose.y),
            (left_eye.x, left_eye.y),
            (right_eye.x, right_eye.y),
        ]
        
        min_distance = float('inf')
        
        for hand_landmarks in hand_landmarks_list:
            # Check all hand points
            for hand_lm in hand_landmarks:
                hand_point = (hand_lm.x, hand_lm.y)
                
                # Distance to closest face point
                for face_point in face_points:
                    dist = np.linalg.norm(
                        np.array(hand_point) - np.array(face_point)
                    )
                    min_distance = min(min_distance, dist)
        
        return min_distance
    
    def _calculate_hand_movement(self, hand_landmarks_list):
        """Calculate hand movement magnitude"""
        if not hand_landmarks_list:
            return 0.0
        
        total_movement = 0.0
        
        for hand_landmarks in hand_landmarks_list:
            # Calculate movement from wrist to fingers
            wrist = hand_landmarks[0]
            
            for finger_tip_idx in [4, 8, 12, 16, 20]:  # Finger tips
                tip = hand_landmarks[finger_tip_idx]
                movement = np.linalg.norm(
                    np.array([tip.x, tip.y]) - np.array([wrist.x, wrist.y])
                )
                total_movement += movement
        
        return total_movement / len(hand_landmarks_list)
    
    def _calculate_head_tilt(self, pose_landmarks):
        """Calculate head tilt angle (left-right rotation)"""
        left_ear = pose_landmarks[3]
        right_ear = pose_landmarks[6]
        
        ear_diff_x = right_ear.x - left_ear.x
        ear_diff_y = right_ear.y - left_ear.y
        
        # Angle from vertical (0 = upright, >0 = tilted)
        angle = np.arctan2(ear_diff_y, ear_diff_x)
        
        self.head_pos_history.append(angle)
        
        # Return magnitude of tilt change
        if len(self.head_pos_history) > 1:
            return abs(self.head_pos_history[-1] - self.head_pos_history[-2])
        
        return 0.0
    
    def _hand_to_landmark_distance(self, hand_landmarks, pose_landmark):
        """Distance from hand to specific pose landmark"""
        min_dist = float('inf')
        
        for hand_lm in hand_landmarks:
            dist = np.linalg.norm(
                np.array([hand_lm.x, hand_lm.y]) - 
                np.array([pose_landmark.x, pose_landmark.y])
            )
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    # ==================== BEHAVIOR DETECTION ====================
    """
    def _detect_head_scratching(self, hand_face_distance, hand_movement):
        #Detect head scratching:
        #- Hand near head region
        #- Moderate movement (scratching motion)

        close_to_face = hand_face_distance < 0.15  # Normalized distance
        moderate_movement = 0.05 < hand_movement < 0.3
        
        if close_to_face and moderate_movement:
            return 0.8
        elif close_to_face:
            return 0.4
        
        return 0.0
    """

    def _detect_nose_scratching(self, pose_landmarks, hand_landmarks_list):
        #Detect nose scratching:
        #- Hand very close to nose
        #- Touching/rubbing motion
        
        if not hand_landmarks_list:
            return 0.0
        
        nose = pose_landmarks[0]
        confidence = 0.0
        
        for hand_lm in hand_landmarks_list:
            # Check if hand fingers are near nose
            index_finger = hand_lm[8]  # Index finger tip
            middle_finger = hand_lm[12]  # Middle finger tip
            
            nose_dist_index = np.linalg.norm(
                np.array([index_finger.x, index_finger.y]) -
                np.array([nose.x, nose.y])
            )
            nose_dist_middle = np.linalg.norm(
                np.array([middle_finger.x, middle_finger.y]) -
                np.array([nose.x, nose.y])
            )
            
            if nose_dist_index < 0.08 or nose_dist_middle < 0.08:
                confidence = max(confidence, 0.9)
            elif nose_dist_index < 0.15 or nose_dist_middle < 0.15:
                confidence = max(confidence, 0.6)
        
        return confidence
    
    """
    def _detect_head_tilting(self, head_tilt):
        #Detect head tilting:
        #- Significant head angle change
        #- Sustained tilt position
        
        # If head tilt angle is significant
        if head_tilt > 0.05:  # Threshold in radians
            # Check history for sustained tilt
            if len(self.head_pos_history) > 5:
                recent_tilts = list(self.head_pos_history)[-5:]
                avg_tilt = np.mean(recent_tilts)
                
                if avg_tilt > 0.05:
                    return 0.7
            
            return 0.5
        
        return 0.0

    def _detect_face_touching(self, hand_face_distance):
        #Detect general face touching:
        #- Hand in immediate face region

        if hand_face_distance < 0.12:
            return 0.8
        elif hand_face_distance < 0.20:
            return 0.5
        
        return 0.0

    def _detect_ear_touching(self, pose_landmarks, hand_landmarks_list):
        
        #Detect ear touching/rubbing:
        # Hand near ear landmarks
        
        if not hand_landmarks_list:
            return 0.0
        
        left_ear = pose_landmarks[3]
        right_ear = pose_landmarks[6]
        
        confidence = 0.0
        
        for hand_lm in hand_landmarks_list:
            # Check hand palm and fingers
            palm = hand_lm[0]  # Wrist/palm
            
            left_ear_dist = np.linalg.norm(
                np.array([palm.x, palm.y]) - np.array([left_ear.x, left_ear.y])
            )
            right_ear_dist = np.linalg.norm(
                np.array([palm.x, palm.y]) - np.array([right_ear.x, right_ear.y])
            )
            
            if left_ear_dist < 0.15 or right_ear_dist < 0.15:
                confidence = max(confidence, 0.7)
        
        return confidence
    """

    def _detect_neck_rubbing(self, pose_landmarks, hand_landmarks_list):
        """
        Detect neck/shoulder rubbing:
        - Hand between neck and shoulders
        - Rubbing motion
        """
        if not hand_landmarks_list:
            return 0.0
        
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        neck_center = pose_landmarks[0]  # Approximate neck area
        
        confidence = 0.0
        
        for hand_lm in hand_landmarks_list:
            palm = hand_lm[0]
            palm_point = np.array([palm.x, palm.y])
            
            shoulder_dist = min(
                np.linalg.norm(palm_point - np.array([left_shoulder.x, left_shoulder.y])),
                np.linalg.norm(palm_point - np.array([right_shoulder.x, right_shoulder.y]))
            )
            
            if shoulder_dist < 0.20:
                confidence = max(confidence, 0.65)
        
        return confidence


    """
    def _detect_hand_wringing(self, hand_landmarks_list):
        
        #Detect hand wringing:
        #- Both hands visible
        #- Hands close together and moving
        
        if len(hand_landmarks_list) < 2:
            return 0.0
        
        hand1 = hand_landmarks_list[0][0]  # First hand palm
        hand2 = hand_landmarks_list[1][0]  # Second hand palm
        
        hand_distance = np.linalg.norm(
            np.array([hand1.x, hand1.y]) - np.array([hand2.x, hand2.y])
        )
        
        # Hands close together
        if hand_distance < 0.15:
            return 0.75
        elif hand_distance < 0.25:
            return 0.5
        
        return 0.0
    """    
    def _detect_fidgeting(self, hand_movement):
        
        #Detect fidgeting:
        #- Continuous small hand movements
        
        if hand_movement > 0.15:
            return 0.6
        elif hand_movement > 0.1:
            return 0.4
        
        return 0.0


    def get_habit_level(self):
        """
        Calculate overall habit level (0.0 = relaxed, 1.0 = highly habited)
        Based on behavior history
        """
        if not self.behavior_history:
            return 0.0
        
        habit_behaviors = [
            #HabitBehavior.HEAD_SCRATCHING,
            HabitBehavior.NOSE_SCRATCHING,
            #HabitBehavior.HEAD_TILTING,
            #HabitBehavior.FIDGETING,
            #HabitBehavior.HAND_WRINGING,
            HabitBehavior.NECK_RUBBING,
        ]
        
        recent_behaviors = list(self.behavior_history)[-5:]
        
        habit_count = sum(
            1 for behavior, confidence in recent_behaviors
            if behavior in habit_behaviors and confidence > 0.5
        )
        
        habit_level = habit_count / len(recent_behaviors) if recent_behaviors else 0.0
        
        return min(habit_level, 1.0)

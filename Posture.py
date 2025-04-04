import cv2
import numpy as np 
import mediapipe as mp 
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class PostureKeyPoints:
    nose: Tuple[float, float, float]
    left_shoulder: Tuple[float, float, float]
    right_shoulder: Tuple[float, float, float]
    left_ear: Tuple[float, float, float]
    right_ear: Tuple[float, float, float]
    left_hip: Tuple[float, float, float]
    right_hip: Tuple[float, float, float]

class FeatureDetection:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode = True,
            model_complexity = 2, 
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5

        )

    def detect_features(self, image) -> PostureKeyPoints:
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            raise ValueError("No pose detected in image")
        landmarks = results.pose_landmarks.landmark

        def normalize_coords(landmark):
            pixelx = landmark.x* width
            pixely = landmark.y* height
            return(pixelx / width, pixely / height, landmark.visibility)


        return PostureKeyPoints(
            nose = normalize_coords(landmarks[self.mp_pose.PoseLandmark.NOSE]),
            left_shoulder = normalize_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]),
            right_shoulder = normalize_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]),
            left_ear = normalize_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]),
            right_ear = normalize_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]),
            left_hip = normalize_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]),
            right_hip = normalize_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]),
        )
    def visualize_landmarks(self, image, keypoints: PostureKeyPoints):
        height, width = image.shape[:2]

        def to_pixel_coords(point):
            return(
                int(point[0]* width),
                int(point[1]* height)
            )
        
        points = [
            (keypoints.nose, "Nose"),
            (keypoints.left_shoulder, "L Shoulder"),
            (keypoints.right_shoulder, "R Shoulder"),
            (keypoints.left_ear, "L Ear"),
            (keypoints.right_ear, "R Ear"),
            (keypoints.left_hip, "L Hip"),
            (keypoints.right_hip, "R Hip"),
         ]
        
        for point, label in points:
            x, y = to_pixel_coords(point)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return image

    
class PostureAnalyzer:
    @staticmethod
    def calculate_angles(keypoints: PostureKeyPoints) -> dict: 
        #calculate neck angles (between ears and shoulders)
        neck_angle = PostureAnalyzer._calculate_neck_angle(
            keypoints.left_ear, keypoints.right_ear,
            keypoints.left_shoulder, keypoints.right_shoulder
        )
        #calculate shoulder angle (between shoulders and hips)
        shoulder_angle = PostureAnalyzer._calculate_shoulder_angle(
            keypoints.left_shoulder, keypoints.right_shoulder,
            keypoints.left_hip, keypoints.right_hip
        )

        #calculate forward head position
        forward_head = PostureAnalyzer._calculate_forward_head(
            keypoints.nose,
            keypoints.left_shoulder, keypoints.right_shoulder
        )

        return{
            'neck_angle': neck_angle,
            'shoulder_angle': shoulder_angle,
            'forward_head': forward_head
        }

    @staticmethod
    def _calculate_neck_angle(left_ear, right_ear, left_shoulder, right_shoulder):
        ear_midpoint = np.array([(left_ear[0] + right_ear[0])/2, (left_ear[1] + right_ear[1])/2])
        shoulder_midpoint = np.array([(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1]+ right_shoulder[1])/2])
        neck_vector = ear_midpoint - shoulder_midpoint
        vertical = np.array([0,-1])

        angle = np.degrees(np.arccos(np.dot(neck_vector,vertical)/ (np.linalg.norm(neck_vector)* np.linalg.norm(vertical))))
        return angle
    @staticmethod
    def _calculate_shoulder_angle(left_shoulder, right_shoulder, left_hip, right_hip):
        shoulder_vector = np.array([(right_shoulder[0] - left_shoulder[0]), right_shoulder[1] - left_shoulder[1]])
        hip_vector = np.array([(right_hip[0] - left_shoulder[0]), right_hip[1] - left_hip[1]])

        angle = np.degrees(np.arccos(np.dot(shoulder_vector, hip_vector)/ (np.linalg.norm(shoulder_vector) * np.linalg.norm(hip_vector))))
        return angle
    @staticmethod
    def _calculate_forward_head(nose, left_shoulder, right_shoulder):
        shoulder_midpoint = np.array([(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2])
        nose_point = np.array([nose[0] , nose[1]])

        #calculate horizontal distance between both midpoints
        forward_distance = nose_point[0] - shoulder_midpoint[0]
        return forward_distance

def analyze_posture_from_image(image_path):

    keypoint_detector = FeatureDetection()

    if image_path is None: 
        print(f"Could not read image")
        return 0
    else: 
        image = cv2.resize(cv2.imread(image_path),(0,0),fx=0.35, fy=0.26)
    
    try:
        keypoints = keypoint_detector.detect_features(image)
        visualize_image = keypoint_detector.visualize_landmarks(image.copy(), keypoints)
        posture_analyzer = PostureAnalyzer()
        angles = posture_analyzer.calculate_angles(keypoints)

        return{
            'keypoints': keypoints,
            'angles': angles,
            'visualized_image': visualize_image
        }
    except Exception as e:
        print(f"Error is Pose detection: {str(e)}")
        return None

def main():
    image_path = "C:/University Work - Year 3/20241111_114655.jpg" #image input here
    results = analyze_posture_from_image(image_path)

    if results:
        cv2.imshow("Pose detection", results['visualized_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Calculated Angles:")
        print(results['angles'])


if __name__ == "__main__":
    main()
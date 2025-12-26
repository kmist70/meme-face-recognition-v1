import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import os
import glob

class MemeMatcher:
    # initializes landmarks
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # key landmarks based on MediaPipe Map
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH_OUTER = [61, 291, 13, 14]
        self.LEFT_BROW = [70, 63, 105, 66, 107]
        self.RIGHT_BROW = [336, 296, 334, 293, 300]
        
        self.database = []

    # gets landmark data and stores it in numpy array
    def get_landmarks(self, image):
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks: return None
        
        # Get the first face found
        landmarks = results.multi_face_landmarks[0]
        points = []
        for lm in landmarks.landmark:
            points.append([lm.x * w, lm.y * h])
            
        return np.array(points)

    # calculates the ratios of features
    def calculate_features(self, landmarks):
        if landmarks is None: return None
        
        def eye_aspect_ratio(eye_points):
            A = dist.euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
            B = dist.euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
            
            C = dist.euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
            return (A + B) / (2.0 * C)

        left_ear = eye_aspect_ratio(self.LEFT_EYE)
        right_ear = eye_aspect_ratio(self.RIGHT_EYE)
        
        mouth_h = dist.euclidean(landmarks[self.MOUTH_OUTER[2]], landmarks[self.MOUTH_OUTER[3]])
        mouth_w = dist.euclidean(landmarks[self.MOUTH_OUTER[0]], landmarks[self.MOUTH_OUTER[1]])
        mar = mouth_h / mouth_w
        
        return [left_ear, right_ear, mar]

    # creates a database of memes and their features
    def init_database(self, folder_path="images"):
        print("Initializing database...")
        image_paths = glob.glob(os.path.join(folder_path, "*.*"))
        
        # goes through each image in the folder
        for path in image_paths:
            img = cv2.imread(path)
            if img is None: continue
            
            landmarks = self.get_landmarks(img)
            features = self.calculate_features(landmarks)
            
            # if the image contains geometric features, the image is added to the database
            if features:
                self.database.append({
                    "path": path,
                    "image": img,
                    "features": features
                })
        
        if len(self.database) == 1:
            print("Loaded 1 image.")
        else:
            print(f"Loaded {len(self.database)} images.")

    # finds the closest meme to match the live face features
    def find_match(self, live_features):
        if not self.database or not live_features:
            return None
                
        best_match = None
        min_dist = float('inf')
        
        for item in self.database:
            # Calculate Euclidean distance between the vectors
            # Lower distance = Better match
            d = dist.euclidean(live_features, item["features"])
                
            if d < min_dist:
                min_dist = d
                best_match = item["image"]
                    
        return best_match

import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from scipy.spatial import distance as dist
from meme_matcher import MemeMatcher

# main loop to match facial features
matcher = MemeMatcher()
matcher.init_database("memes")
cam = cv2.VideoCapture(1)   # 1 is the laptop webcam

while True:
    ret, frame = cam.read()
    if not ret: break
    
    # process the current frame
    landmarks = matcher.get_landmarks(frame)
    live_features = matcher.calculate_features(landmarks)
    
    # finds the best match for the current frame and resizes for display if found
    match_img = matcher.find_match(live_features)
    display_frame = cv2.resize(frame, (1280, 720))
    
    if match_img is not None:
        match_resized = cv2.resize(match_img, (1280, 720))
        combined = np.hstack((display_frame, match_resized))
    else: combined = display_frame
    
    cv2.imshow("Meme Matcher LIVE", combined)
    
    if cv2.waitKey(5) & 0xFF == 27: # Press ESC to quit
        break

cam.release()
cv2.destroyAllWindows()

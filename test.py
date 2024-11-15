import cv2
import mediapipe as mp
import numpy as np
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
import winsound  # For sound playback on Windows

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the image poses from the folder
pose_image_folder = './pose_images'
pose_image_paths = [
    os.path.join(pose_image_folder, f)
    for f in os.listdir(pose_image_folder)
    if f.endswith('.jpg') or f.endswith('.png')
]

# Error tolerance can be adjusted here
similarity_threshold = 0.975
game_duration = 100  # Total game duration in seconds
hold_time = 1  # Hold time for each pose in seconds
key_landmarks_indices = [
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value
]

# Function to extract pose landmarks from an image
def extract_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        try:
            landmarks = [(lm.x, lm.y, lm.z) for i, lm in enumerate(results.pose_landmarks.landmark) if i in key_landmarks_indices]
            return np.array(landmarks)
        except IndexError:
            print("Not enough landmarks detected in image.")
            return None
    return None

def normalize_landmarks(landmarks):
    if landmarks is None:
        return None
    
    left_hip = landmarks[key_landmarks_indices.index(mp_pose.PoseLandmark.LEFT_HIP.value)]
    right_hip = landmarks[key_landmarks_indices.index(mp_pose.PoseLandmark.RIGHT_HIP.value)]
    hip_center = (left_hip + right_hip) / 2

    normalized_landmarks = landmarks - hip_center
    
    shoulder_distance = np.linalg.norm(landmarks[key_landmarks_indices.index(mp_pose.PoseLandmark.LEFT_SHOULDER.value)] - 
                                       landmarks[key_landmarks_indices.index(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)])
    
    if shoulder_distance > 0:
        normalized_landmarks /= shoulder_distance
    
    return normalized_landmarks

def compare_poses(target_pose, current_pose):
    if target_pose is None or current_pose is None:
        return False

    target_normalized = normalize_landmarks(target_pose)
    current_normalized = normalize_landmarks(current_pose)

    if target_normalized is None or current_normalized is None:
        return False

    similarity = cosine_similarity([target_normalized.flatten()], [current_normalized.flatten()])[0][0]
    return similarity >= similarity_threshold

# Function to play a sound when a pose is achieved
def play_achievement_sound():
    duration = 500  # milliseconds
    frequency = 1000  # Hz
    winsound.Beep(frequency, duration)

# Function to display a start screen
def display_start_screen():
    start_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(start_screen, "Press Enter to Start the Game", (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.namedWindow('Pose Tracker', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Pose Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        cv2.imshow('Pose Tracker', start_screen)
        key = cv2.waitKey(10) & 0xFF
        if key == 13:  # Enter key
            break

# Function to display a countdown
def display_countdown():
    countdown_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(3, 0, -1):
        countdown_screen.fill(0)
        cv2.putText(countdown_screen, str(i), (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow('Pose Tracker', countdown_screen)
        cv2.waitKey(1000)

# Main gameplay loop
def play_pose_game():
    display_start_screen()  # Show the start screen
    display_countdown()     # Show the countdown

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    score = 0
    current_image_index = 0
    target_pose = extract_pose_landmarks(pose_image_paths[current_image_index])
    start_time = time.time()
    hold_start_time = None
    pose_held = False

    while cap.isOpened():
        elapsed_time = time.time() - start_time
        remaining_time = int(game_duration - elapsed_time)

        if remaining_time <= 0:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            current_pose = np.array([(lm.x, lm.y, lm.z) for i, lm in enumerate(results.pose_landmarks.landmark) if i in key_landmarks_indices])

            if compare_poses(target_pose, current_pose):
                if not pose_held:
                    hold_start_time = time.time()
                    pose_held = True
                elif time.time() - hold_start_time >= hold_time:
                    score += 1
                    play_achievement_sound()
                    current_image_index = (current_image_index + 1) % len(pose_image_paths)
                    target_pose = extract_pose_landmarks(pose_image_paths[current_image_index])
                    pose_held = False
            else:
                pose_held = False

        overlay = frame.copy()
        cv2.putText(overlay, f"Score: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Time Left: {remaining_time}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Pose Tracker', overlay)

        if cv2.waitKey(10) & 0xFF == 27:  # ESC key to quit
            break

    cap.release()
    cv2.destroyAllWindows()

play_pose_game()

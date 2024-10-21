import cv2
import mediapipe as mp
import numpy as np
import os
import random
import time
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the image poses from the folder
pose_image_folder = './pose_images'
pose_image_paths = [os.path.join(pose_image_folder, f) for f in os.listdir(pose_image_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Error tolerance can be adjusted here
similarity_threshold = 0.95  # Increase similarity threshold to make the game harder

# Game duration in seconds
game_duration = 100

# List of key landmarks to track (excluding face, hands, feet, etc.)
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

# Check if the pose_image_paths list is not empty
if not pose_image_paths:
    raise ValueError("No pose images found in the specified folder. Please check './pose_images' for valid images.")

# Function to extract pose landmarks from an image
def extract_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        try:
            # Only extract key landmarks (e.g., exclude face landmarks)
            landmarks = [(lm.x, lm.y, lm.z) for i, lm in enumerate(results.pose_landmarks.landmark) if i in key_landmarks_indices]
            return np.array(landmarks)
        except IndexError:
            print("Not enough landmarks detected in image.")
            return None
    return None

# Normalize the pose landmarks by scaling them relative to key body parts (hips or shoulders)
def normalize_landmarks(landmarks):
    if landmarks is None:
        return None
    
    # Normalize coordinates relative to the center of the hips (midpoint of left and right hips)
    left_hip = landmarks[key_landmarks_indices.index(mp_pose.PoseLandmark.LEFT_HIP.value)]
    right_hip = landmarks[key_landmarks_indices.index(mp_pose.PoseLandmark.RIGHT_HIP.value)]
    hip_center = (left_hip + right_hip) / 2

    # Subtract the hip center to make the pose relative to the body
    normalized_landmarks = landmarks - hip_center

    # Optionally normalize by scale (e.g., distance between shoulders) to account for body size
    shoulder_distance = np.linalg.norm(landmarks[key_landmarks_indices.index(mp_pose.PoseLandmark.LEFT_SHOULDER.value)] - 
                                       landmarks[key_landmarks_indices.index(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)])
    
    if shoulder_distance > 0:
        normalized_landmarks /= shoulder_distance
    
    return normalized_landmarks

# Compare similarity of poses by directly checking landmark positions
def compare_poses(target_pose, current_pose):
    if target_pose is None or current_pose is None:
        return False

    # Normalize both poses
    target_normalized = normalize_landmarks(target_pose)
    current_normalized = normalize_landmarks(current_pose)

    if target_normalized is None or current_normalized is None:
        return False  # Skip comparison if any normalized pose is invalid

    # Calculate cosine similarity between the two normalized poses
    similarity = cosine_similarity([target_normalized.flatten()], [current_normalized.flatten()])[0][0]

    # Check if similarity exceeds the (stricter) threshold
    return similarity >= similarity_threshold

# Function to draw the UI overlay
def draw_ui_overlay(frame, score, total_poses, time_remaining):
    # Dimensions of the overlay box
    box_width = 350
    box_height = 150
    box_color = (255, 255, 255)  # White box
    text_color = (0, 0, 0)  # Black text

    # Create a rounded rectangle for the overlay
    overlay = frame.copy()
    top_left_corner = (10, 10)
    bottom_right_corner = (top_left_corner[0] + box_width, top_left_corner[1] + box_height)

    # Add a rounded rectangle
    cv2.rectangle(overlay, top_left_corner, bottom_right_corner, box_color, -1, cv2.LINE_AA)
    
    # Add rounded corners by drawing circles at the corners
    radius = 20
    cv2.circle(overlay, (top_left_corner[0] + radius, top_left_corner[1] + radius), radius, box_color, -1)
    cv2.circle(overlay, (bottom_right_corner[0] - radius, top_left_corner[1] + radius), radius, box_color, -1)
    cv2.circle(overlay, (top_left_corner[0] + radius, bottom_right_corner[1] - radius), radius, box_color, -1)
    cv2.circle(overlay, (bottom_right_corner[0] - radius, bottom_right_corner[1] - radius), radius, box_color, -1)
    
    # Add score, total poses, and time remaining to the overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, f"Poses: {score}/{total_poses}", (30, 50), font, 1, text_color, 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Score: {score}", (30, 90), font, 1, text_color, 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Time: {time_remaining}s", (30, 130), font, 1, text_color, 2, cv2.LINE_AA)
    
    # Blend the overlay with the original frame
    alpha = 1.0
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame

# Function to display the final score and restart message on a black screen
def display_end_screen(score):
    black_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Black screen
    message = f"Final Score: {score}"
    restart_message = "Press Enter to Play Again or ESC to Exit"
    
    # Display the score
    cv2.putText(black_screen, message, (600, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
    
    # Display the restart message
    cv2.putText(black_screen, restart_message, (500, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the black screen and wait for user input
    while True:
        cv2.imshow('Pose Tracker', black_screen)
        key = cv2.waitKey(10)
        
        # If "Enter" is pressed, exit and restart the game
        if key == 13:  # ASCII code for Enter is 13
            break
        # If "ESC" is pressed, exit the game
        if key == 27:  # ASCII code for ESC is 27
            cv2.destroyAllWindows()
            exit()

# Main game function
def play_pose_game():
    while True:
        cap = cv2.VideoCapture(1)
        
        # Set camera resolution to 1920x1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        score = 0
        current_image_index = 0
        target_pose = extract_pose_landmarks(pose_image_paths[current_image_index])

        # Start time for the game
        start_time = time.time()

        while cap.isOpened():
            elapsed_time = time.time() - start_time
            remaining_time = int(game_duration - elapsed_time)

            # End the game if time is up
            if remaining_time <= 0:
                cap.release()
                cv2.destroyAllWindows()
                display_end_screen(score)
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for Mediapipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract current pose landmarks from the webcam, using only key landmarks
                current_pose = np.array([(lm.x, lm.y, lm.z) for i, lm in enumerate(results.pose_landmarks.landmark) if i in key_landmarks_indices])

                # Check if the current pose matches the target pose
                if compare_poses(target_pose, current_pose):
                    score += 1
                    current_image_index += 1

                    if current_image_index >= len(pose_image_paths):
                        current_image_index = 0  # Loop back to the first image

                    # Load the next target pose image
                    target_pose = extract_pose_landmarks(pose_image_paths[current_image_index])

            # Draw the UI overlay
            frame = draw_ui_overlay(frame, score, len(pose_image_paths), remaining_time)

            # Display the webcam feed
            cv2.imshow('Pose Tracker', frame)

            # Exit the game if 'ESC' is pressed
            if cv2.waitKey(10) == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

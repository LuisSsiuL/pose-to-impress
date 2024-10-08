import cv2
import mediapipe as mp
import numpy as np
import os
import random
import time

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the image poses from the folder
pose_image_folder = './pose_images'
output_image_folder = './output_images'

pose_image_paths = [os.path.join(pose_image_folder, f) for f in os.listdir(pose_image_folder) if f.endswith('.jpg') or f.endswith('.png')]
output_image_paths = [os.path.join(output_image_folder, f) for f in os.listdir(output_image_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Error tolerance can be adjusted here
error_tolerance = 0.15  # Increased tolerance to 15%

# Game duration in seconds
game_duration = 100

# Function to extract pose landmarks from image
def extract_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        return np.array(landmarks)
    return None

# Normalize the pose landmarks by scaling them relative to key body parts (hips or shoulders)
def normalize_landmarks(landmarks):
    if landmarks is None:
        return None
    
    # Normalize coordinates relative to the center of the hips (midpoint of left and right hips)
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_center = (left_hip + right_hip) / 2

    # Subtract the hip center to make the pose relative to the body
    normalized_landmarks = landmarks - hip_center

    # Optionally normalize by scale (e.g., distance between shoulders) to account for body size
    shoulder_distance = np.linalg.norm(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] - 
                                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    
    if shoulder_distance > 0:
        normalized_landmarks /= shoulder_distance
    
    return normalized_landmarks

# Compare similarity of poses with improved tolerance
def compare_poses(target_pose, current_pose):
    if target_pose is None or current_pose is None:
        return False

    # Normalize both poses to make the comparison structure-based
    normalized_target_pose = normalize_landmarks(target_pose)
    normalized_current_pose = normalize_landmarks(current_pose)

    # Compare relative distances between keypoints, not just coordinates
    relative_distances_target = np.linalg.norm(normalized_target_pose, axis=1)
    relative_distances_current = np.linalg.norm(normalized_current_pose, axis=1)

    # Calculate the difference and allow a tolerance
    error = np.abs(relative_distances_target - relative_distances_current)
    avg_error = np.mean(error)

    return avg_error <= error_tolerance

# Function to display the final score and restart message on a black screen
def display_end_screen(score):
    black_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Black screen
    message = f"Final Score: {score}"
    restart_message = "Press Enter to Play Again"
    
    # Display the score
    cv2.putText(black_screen, message, (600, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
    
    # Display the restart message
    cv2.putText(black_screen, restart_message, (550, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the black screen
    while True:
        cv2.imshow('Pose Tracker', black_screen)

        # If "Enter" is pressed, exit and restart the game
        if cv2.waitKey(10) == 13:  # ASCII code for Enter is 13
            break

def play_pose_game():
    while True:
        cap = cv2.VideoCapture(0)
        
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
                # Extract current pose landmarks from the webcam
                current_pose = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])

                # Compare poses with the improved tolerance function based on structure
                if compare_poses(target_pose, current_pose):
                    score += 1
                    print(f"Pose achieved! Current Score: {score}")
                    current_image_index = (current_image_index + 1) % len(pose_image_paths)
                    target_pose = extract_pose_landmarks(pose_image_paths[current_image_index])

            # Draw landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the current target pose image in the top right corner
            current_pose_img = cv2.imread(pose_image_paths[current_image_index])
            current_pose_img = cv2.resize(current_pose_img, (200, 200))
            frame_height, frame_width, _ = frame.shape
            frame[0:200, frame_width-200:frame_width] = current_pose_img  # Place in top right corner

            # Display the corresponding outline image from './output_images'
            current_output_img = cv2.imread(output_image_paths[current_image_index])
            current_output_img = cv2.resize(current_output_img, (200, 200))
            frame[200:400, frame_width-200:frame_width] = current_output_img  # Place just below the target pose

            # Display the score on the frame
            cv2.putText(frame, f"Score: {score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the remaining time on the frame
            cv2.putText(frame, f"Time Left: {remaining_time}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the current frame with tracking
            cv2.imshow('Pose Tracker', frame)

            # Break the loop with 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    play_pose_game()

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
import winsound  # For sound playback on Windows
import pygame


pygame.mixer.init()
# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the image poses from the folder
pose_image_folder = './pose_images'
pose_image_paths = [os.path.join(pose_image_folder, f) for f in os.listdir(pose_image_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Error tolerance can be adjusted here
similarity_threshold = 0.975
game_duration = 100
hold_time = 1  # Hold time in seconds
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

def play_audio(file_path):
    """
    Plays an MP3 file.
    """
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

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

import cv2
import os

def add_frame_to_images(input_folder, frame_path, output_folder):
    """
    Overlays a frame image on top of all images in a folder and saves the results.

    Parameters:
    - input_folder: Path to the folder containing input images.
    - frame_path: Path to the frame image (1280x720).
    - output_folder: Path to save the edited images.
    """
    # Load the frame image
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if it exists

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Skip non-image files
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        # Load the screenshot
        screenshot = cv2.imread(input_path)
        if screenshot is None:
            print(f"Skipping invalid image file: {filename}")
            continue

        # Resize screenshot to 1280x720 if needed
        screenshot = cv2.resize(screenshot, (1280, 720))

        # If the frame has an alpha channel, blend using the alpha mask
        if frame.shape[2] == 4:  # Check if frame has an alpha channel
            alpha_channel = frame[:, :, 3] / 255.0
            for c in range(3):  # Blend each color channel
                screenshot[:, :, c] = screenshot[:, :, c] * (1 - alpha_channel) + frame[:, :, c] * alpha_channel
        else:
            # If no alpha channel, simply overlay the frame
            screenshot = cv2.addWeighted(screenshot, 1.0, frame, 1.0, 0)

        # Save the edited image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, screenshot)
        print(f"Edited image saved to {output_path}")



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
    # Play a sound when a pose is achieved
    play_audio("correct.mp3")

# Function to display pause menu
def show_pause_menu():
    pause_menu = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(pause_menu, "Paused", (800, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(pause_menu, "Press 'r' to Resume", (750, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pause_menu, "Press 's' to Restart", (750, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pause_menu, "Press 'e' to Exit", (750, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.namedWindow('Pose Tracker', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Pose Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        cv2.imshow('Pose Tracker', pause_menu)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            return 'resume'
        elif key == ord('s'):
            return 'restart'
        elif key == ord('e'):
            cv2.destroyAllWindows()
            exit()

# Function to display the final score and end screen in fullscreen
def display_end_screen(score):
    end_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    message = f"Final Score: {score}"
    cv2.putText(end_screen, message, (800, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(end_screen, "Press 's' to Restart", (750, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(end_screen, "Press 'e' to Exit", (750, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow('Pose Tracker', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Pose Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    add_frame_to_images('./screenshot_images', './frame.png', './final_images')
    while True:
        cv2.imshow('Pose Tracker', end_screen)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            return 'restart'
        elif key == ord('e'):
            cv2.destroyAllWindows()
            exit()

def draw_overlay(frame, score, remaining_time):
    overlay = frame.copy()
    box_width, box_height, box_x, box_y = 300, 100, 10, 10
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), 2)
    cv2.putText(overlay, f"Score: {score}", (box_x + 20, box_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Time Left: {remaining_time}", (box_x + 20, box_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return overlay

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
    play_audio('start.mp3')
    for i in range(3, 0, -1):
        countdown_screen.fill(0)
        cv2.putText(countdown_screen, str(i), (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow('Pose Tracker', countdown_screen)
        cv2.waitKey(1000)

def play_pose_game():
    display_start_screen()  # Show the start screen
    display_countdown()     # Show the countdown
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    score = 0
    current_image_index = 0
    target_pose = extract_pose_landmarks(pose_image_paths[current_image_index])
    start_time = time.time()
    hold_start_time = None
    pose_held = False
    paused = False

    cv2.namedWindow('Pose Tracker', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Pose Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Feed', 640, 360)
    
    while cap.isOpened():
        if paused:
            action = show_pause_menu()
            if action == 'resume':
                paused = False
            elif action == 'restart':
                display_start_screen()  # Show the start screen
                display_countdown()     # Show the countdown
                score = 0
                start_time = time.time()
                current_image_index = 0
                target_pose = extract_pose_landmarks(pose_image_paths[current_image_index])
                paused = False
            continue

        elapsed_time = time.time() - start_time
        remaining_time = int(game_duration - elapsed_time)
        
        if remaining_time <= 0:
            action = display_end_screen(score)
            if action == 'restart':
                # display_countdown()  
                score = 0
                start_time = time.time()
                current_image_index = 0
                target_pose = extract_pose_landmarks(pose_image_paths[current_image_index])
                display_start_screen()
                display_countdown()  
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            current_pose = np.array([(lm.x, lm.y, lm.z) for i, lm in enumerate(results.pose_landmarks.landmark) if i in key_landmarks_indices])

            # Check if current pose matches the target pose
            if compare_poses(target_pose, current_pose):
                if not pose_held:
                    hold_start_time = time.time()
                    pose_held = True
                elif time.time() - hold_start_time >= hold_time:
                    score += 2
                    play_achievement_sound()
                    
                    # Take a screenshot after holding the pose
                    screenshot_path = f"./screenshot_images/screenshot_{current_image_index + 1}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Pose achieved! Screenshot saved as {screenshot_path}")
                    
                    # Move to the next pose
                    current_image_index = (current_image_index + 1) % len(pose_image_paths)
                    target_pose = extract_pose_landmarks(pose_image_paths[current_image_index])
                    pose_held = False
            else:
                pose_held = False  # Reset if pose is not held continuously

        # mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the current target pose image in the top right corner
        current_pose_img = cv2.imread(pose_image_paths[current_image_index])
        current_pose_img = cv2.resize(current_pose_img, (426, 240))  # Resize to one-third of 1280 x 720
        frame_height, frame_width, _ = frame.shape
        frame[0:240, frame_width-426:frame_width] = current_pose_img  # Place in the top right corner

        overlayed_frame = draw_overlay(frame, score, remaining_time)
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Pose Tracker', overlayed_frame)

        cv2.imshow('Camera Feed', frame)
        
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # 'Esc' key to pause
            paused = True
        elif key == ord('s'):  # 'S' key to skip pose
            print("Pose skipped!")
            current_image_index = (current_image_index + 1) % len(pose_image_paths)
            target_pose = extract_pose_landmarks(pose_image_paths[current_image_index])
            pose_held = False

    cap.release()
    cv2.destroyAllWindows()

play_pose_game()

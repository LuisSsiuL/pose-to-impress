import cv2
import mediapipe as mp
import os
import json

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Paths for input, output, and JSON files
input_folder = './pose_images'
output_folder = './output_images'
output_file = './pose_coordinates.json'

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List of key landmarks to keep (nose, shoulders, elbows, wrists, hips, knees, ankles)
key_landmarks_indices = [
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value
]

# Function to normalize landmarks based on hips and shoulder distance
def normalize_landmarks(landmarks, image_shape):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_center = ((left_hip['x'] + right_hip['x']) / 2, (left_hip['y'] + right_hip['y']) / 2)

    # Normalize coordinates relative to hip center
    normalized_landmarks = []
    for lm in landmarks:
        norm_lm = {
            'x': (lm['x'] - hip_center[0]) * image_shape[1],  # Scale by image width
            'y': (lm['y'] - hip_center[1]) * image_shape[0],  # Scale by image height
            'z': lm['z'],  # Z-axis is optional, depending on 3D calculations
            'visibility': lm['visibility']
        }
        normalized_landmarks.append(norm_lm)

    return normalized_landmarks

# Function to process each image: extract landmarks, save JSON, and draw pose
def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = []
        # Extract only the selected landmarks and store them in JSON format
        for i in key_landmarks_indices:
            lm = results.pose_landmarks.landmark[i]
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })

        # Normalize landmarks based on hips for consistency
        normalized_landmarks = normalize_landmarks(landmarks, image.shape)

        # Manually draw only the key landmarks and connections
        for lm in normalized_landmarks:
            # Draw circles for key points
            cv2.circle(image, (int(lm['x']), int(lm['y'])), 5, (0, 255, 0), -1)

        # Draw lines (connections) between key points
        connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
            (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
            (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
            (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
            (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
            (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value)
        ]

        for connection in connections:
            start_idx, end_idx = connection
            start = normalized_landmarks[start_idx]
            end = normalized_landmarks[end_idx]
            start_coords = (int(start['x']), int(start['y']))
            end_coords = (int(end['x']), int(end['y']))
            cv2.line(image, start_coords, end_coords, (0, 255, 0), 2)

        # Save outlined image
        output_image_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_image_path, image)
        print(f"Saved outlined image: {output_image_path}")
        
        return landmarks
    else:
        print(f"No pose detected for: {image_path}")
        return None

# Process all images in the input folder
pose_data = {}
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    landmarks = process_image(image_path)

    if landmarks is not None:
        pose_data[image_name] = landmarks

# Save extracted pose coordinates to a JSON file
with open(output_file, 'w') as f:
    json.dump(pose_data, f)

print(f"Pose data extracted and saved to {output_file}")
print("Processing complete!")

import cv2
import mediapipe as mp
import os
import json

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Paths to input, output folders, and output JSON file.
input_folder = './pose_images'
output_folder = './output_images'
output_file = './pose_coordinates.json'

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Extract pose landmarks from images and save processed images.
pose_data = {}
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is not None:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            # Extract all pose landmarks (33 total)
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                })
            pose_data[image_name] = landmarks

            # Draw landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # Key points
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Connections
            )

            # Save the processed image with landmarks
            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, image)
            print(f"Processed image saved: {output_image_path}")
        else:
            print(f"No pose landmarks found for {image_name}")

# Save the extracted pose coordinates to a JSON file.
with open(output_file, 'w') as f:
    json.dump(pose_data, f)

print(f"Pose data extracted and saved to {output_file}")

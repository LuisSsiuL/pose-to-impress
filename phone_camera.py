import cv2
import numpy as np

# Access IVcam (use index 0 or 1 depending on which camera is connected)
cap = cv2.VideoCapture(1)  # Try 1 if 0 doesn't work

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Process the frame (e.g., convert to grayscale, edge detection, etc.)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Display the frame with edges
    cv2.imshow('IVcam -5 Edge Detection', edges)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import os
import cv2
from datetime import datetime

# Ask user name
user_name = input("Enter the user's name: ")

# Create image folders
user_folder = f"users_image/{user_name}"
os.makedirs(user_folder, exist_ok=True)

# Set up screen capture
screen_capture = cv2.VideoCapture(0)  # 0 represents the default camera

# Control the frames per second
images_per_second = 2  # You can change this value as needed

while True:
    ret, frame = screen_capture.read()
    if not ret:
        break

    # Show real-time video (optional)
    cv2.imshow("Screen Recording", frame)

    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create unique image name based on user and timestamp
    image_name = f"{user_folder}/{user_name}_{current_time}.jpg"

    # Save image
    cv2.imwrite(image_name, frame)

    # Stop recording when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
screen_capture.release()
cv2.destroyAllWindows()

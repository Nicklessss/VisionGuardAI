import os
import cv2

# Ask user name
user_name = input("Ingresa el nombre del usuario: ")

# create image folders
user_folder = f"users_image/{user_name}"
os.makedirs(user_folder, exist_ok=True)
# Set up screen capture
screen_capture = cv2.VideoCapture(0)  # 0 represents the default camera

# Counter for image file names
image_count = 0

# Control the frames per second
images_per_second = 2  # You can change this value as needed

while True:
    ret, frame = screen_capture.read()
    if not ret:
        break

    # Show real-time video (optional)
    cv2.imshow("Screen Recording", frame)

    # Save image if the desired images per second rate is reached
    if image_count % int(30 / images_per_second) == 0:
        cv2.imwrite(f"{user_folder}/frame_{image_count}.jpg", frame)

    image_count += 1

    # Stop recording when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
screen_capture.release()
cv2.destroyAllWindows()

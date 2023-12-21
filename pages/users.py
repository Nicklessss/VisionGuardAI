import streamlit as st
import os
import cv2
from datetime import datetime


def main():
    st.title("Real-time Image Capture")

    # Ask user name
    user_name = st.text_input("Enter the user's name:")

    start_button = st.button("Start Capture")
    stop_button = st.button("Stop Capture", key="stop_button")

    if start_button:
        if user_name:
            # Create image folders
            user_folder = f"users_image/{user_name}"
            os.makedirs(user_folder, exist_ok=True)

            # Set up screen capture
            screen_capture = cv2.VideoCapture(0)  # 0 represents the default camera

            # Control the frames per second
            images_per_second = 2  # You can change this value as needed

            # Create placeholders for displaying video and captured images
            video_placeholder = st.empty()
            images_placeholder = st.empty()

            while True:
                ret, frame = screen_capture.read()
                if not ret:
                    break

                # Show real-time video
                video_placeholder.image(frame, channels="BGR")

                # Get current date and time
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # Create unique image name based on user and timestamp
                image_name = f"{user_folder}/{user_name}_{current_time}.jpg"

                # Save image
                cv2.imwrite(image_name, frame)

                # Stop recording when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q") or stop_button:
                    break

            # Release resources
            screen_capture.release()
            cv2.destroyAllWindows()

            # Show message after capturing images and registering the user
            st.write("Images captured and user registered successfully.")


if __name__ == "__main__":
    main()

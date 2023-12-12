import os
import cv2
# Ask user name
user_name = input("Ingresa el nombre del usuario: ")

# create image folders
user_folder = f"users_image/{user_name}"
os.makedirs(user_folder, exist_ok=True)

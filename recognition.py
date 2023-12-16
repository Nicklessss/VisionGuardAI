# Import Libraries
import face_recognition
import os
import numpy as np

# Load and encode faces
def encode_faces(folder):
    encoded_faces = []
    for filename in os.listdir(folder):
        # Construct full file path
        file_path = os.path.join(folder, filename)
        # Load image file
        image = face_recognition.load_image_file(file_path)
        # Attempt to encode the face
        encodings = face_recognition.face_encodings(image)
        # Check if at least one face was found
        if len(encodings) > 0:
            # Add the first encoding found (assuming one face per image)
            encoded_faces.append(encodings[0])
        else:
            # Print an error message if no faces were found
            print(f"No faces were found in this image: {file_path}")
    return encoded_faces


# Encode Faces
javi_encodings = encode_faces('faces/Javi')
david_encodings = encode_faces('faces/David')
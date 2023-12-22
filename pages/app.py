import streamlit as st
import cv2
import face_recognition
from utils.recognition import javi_encodings, david_encodings


# Function to process video frames for face detection and recognition
def process_frame(frame):
    # Load the face detection model
    net = cv2.dnn.readNetFromCaffe(
        "models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel"
    )

    # Model parameters
    anchonet, altonet = 300, 300
    media = [104, 117, 123]
    umbral = 0.7

    frame = cv2.flip(frame, 1)
    altoframe, anchoframe = frame.shape[:2]

    # Preprocess the image
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (anchonet, altonet), media, swapRB=False, crop=False
    )
    net.setInput(blob)
    detecciones = net.forward()

    # Iterate over each detection
    for i in range(detecciones.shape[2]):
        conf_detect = detecciones[0, 0, i, 2]
        if conf_detect > umbral:
            xmin = int(detecciones[0, 0, i, 3] * anchoframe)
            ymin = int(detecciones[0, 0, i, 4] * altoframe)
            xmax = int(detecciones[0, 0, i, 5] * anchoframe)
            ymax = int(detecciones[0, 0, i, 6] * altoframe)

            # Draw the rectangle around each face
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            # Face recognition
            face_locations = [
                (ymin, xmax, ymax, xmin)
            ]  # Convert to face_recognition format
            current_encodings = face_recognition.face_encodings(frame, face_locations)

            access_label = "ACCESO DENEGADO"
            recognized_name = "Unknown"  # Default name if no match is found
            for encoding in current_encodings:
                javi_results = face_recognition.compare_faces(javi_encodings, encoding)
                david_results = face_recognition.compare_faces(
                    david_encodings, encoding
                )

                if True in javi_results:
                    recognized_name = "Javi"
                    access_label = "ACCESO PERMITIDO"
                elif True in david_results:
                    recognized_name = "David"
                    access_label = "ACCESO PERMITIDO"

            # Display the access label and the name on the frame
            label_position = (xmin, ymax + 20)
            name_position = (xmin, ymax + 45)  # Position for the recognized name
            cv2.putText(
                frame,
                access_label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                recognized_name,
                name_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    return frame


# Streamlit App Interface
st.title("Face Recognition System")

# Button to start and stop the webcam
start_button = st.button("Start Webcam")
stop_button = st.button("Stop Webcam")

# Placeholder for the video frame
frame_placeholder = st.empty()

# Capturing webcam stream
if start_button:
    cap = cv2.VideoCapture(0)  # 0 is typically the ID for the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        frame = process_frame(frame)

        # Convert to RGB and display in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

        if stop_button:
            break

    cap.release()
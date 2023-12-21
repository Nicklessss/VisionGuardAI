import streamlit as st
import cv2
import face_recognition
from utils.recognition import javi_encodings, david_encodings


def load_detection_model():
    return cv2.dnn.readNetFromCaffe(
        "models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel"
    )


def detect_faces(frame, detection_model):
    anchonet, altonet = 300, 300
    media = [104, 117, 123]
    umbral = 0.7

    frame = cv2.flip(frame, 1)
    altoframe, anchoframe = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (anchonet, altonet), media, swapRB=False, crop=False
    )
    detection_model.setInput(blob)
    detecciones = detection_model.forward()

    faces = []
    for i in range(detecciones.shape[2]):
        conf_detect = detecciones[0, 0, i, 2]
        if conf_detect > umbral:
            xmin = int(detecciones[0, 0, i, 3] * anchoframe)
            ymin = int(detecciones[0, 0, i, 4] * altoframe)
            xmax = int(detecciones[0, 0, i, 5] * anchoframe)
            ymax = int(detecciones[0, 0, i, 6] * altoframe)
            faces.append((xmin, ymin, xmax, ymax))

    return faces


def recognize_faces(frame, faces):
    for face in faces:
        (xmin, ymin, xmax, ymax) = face
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        face_locations = [(ymin, xmax, ymax, xmin)]
        current_encodings = face_recognition.face_encodings(frame, face_locations)

        access_label = "ACCESO DENEGADO"
        recognized_name = "Unknown"

        for encoding in current_encodings:
            javi_results = face_recognition.compare_faces(javi_encodings, encoding)
            david_results = face_recognition.compare_faces(david_encodings, encoding)

            if True in javi_results:
                recognized_name = "Javi"
                access_label = "ACCESO PERMITIDO"
            elif True in david_results:
                recognized_name = "David"
                access_label = "ACCESO PERMITIDO"

        label_position = (xmin, ymax + 20)
        name_position = (xmin, ymax + 45)

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


def start_webcam():
    st.title("Face Recognition System")
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    frame_placeholder = st.empty()

    if start_button:
        cap = cv2.VideoCapture(0)

        detection_model = load_detection_model()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces(frame, detection_model)
            frame = recognize_faces(frame, faces)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

            if stop_button:
                break

        cap.release()


start_webcam()

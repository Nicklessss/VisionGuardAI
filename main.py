# Import necessary libraries
import cv2
import face_recognition
from utils.recognition import javi_encodings, david_encodings

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Path to models
prototxt_path = "models/deploy.prototxt"
caffemodel_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Load the face detection model with the correct paths
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Model parameters
anchonet = 300
altonet = 300
media = [104, 117, 123]
umbral = 0.7

# Start processing the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    altoframe = frame.shape[0]
    anchoframe = frame.shape[1]

    # Preprocess the image for the neural network
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
            for encoding in current_encodings:
                javi_results = face_recognition.compare_faces(javi_encodings, encoding)
                david_results = face_recognition.compare_faces(
                    david_encodings, encoding
                )

                if True in javi_results or True in david_results:
                    access_label = "ACCESO PERMITIDO"

            # Display the access label on the frame
            cv2.putText(
                frame,
                access_label,
                (xmin, ymax + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # Display the result
    cv2.imshow("DETECCION DE ROSTROS", frame)

    # Break the loop with the 'Esc' key
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

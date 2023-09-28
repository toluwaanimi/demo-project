import sys

import cv2
import time
import imutils
from imutils.video import VideoStream
import numpy as np
import math

"""SETTINGS AND VARIABLES ________________________________________________________________"""

RASPBERRY_BOOL = False
# If this is run on a linux system, a picamera will be used.
# if sys.platform == "linux":
#     import picamera
#     from picamera.array import PiRGBArray
#
#     RASPBERRY_BOOL = True

# Path to the face-detection model:
pretrained_model = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt",
                                            "models/face_recognition.caffemodel")

video_resolution = (700, 400)  # resolution the video capture will be resized to
video_midpoint = (int(video_resolution[0] / 2),
                  int(video_resolution[1] / 2))

vs = VideoStream(src=0,
                 usePiCamera=RASPBERRY_BOOL,
                 resolution=video_resolution,
                 framerate=13).start()
time.sleep(0.2)

"""FUNCTIONS _____________________________________________________________________________"""


def find_faces_dnn(image):
    """
    Finds human faces in the frame captured by the camera and returns the positions.
    Uses the pretrained model located at pretrained_model.

    Input:
        image: frame captured by the camera

    Return Values:
        face_centers: list of center positions of all detected faces
            list of lists with 2 values (x and y)
        frame: new frame resized with boxes and probabilities drawn around all faces
    """

    frame = imutils.resize(image, width=video_resolution[0])

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    pretrained_model.setInput(blob)
    detections = pretrained_model.forward()
    face_centers = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.4:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        face_center = (int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2))
        position_from_center = (face_center[0] - video_midpoint[0], face_center[1] - video_midpoint[1])
        face_centers.append(position_from_center)

        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.line(frame, video_midpoint, face_center, (0, 200, 0), 5)
        cv2.circle(frame, face_center, 4, (0, 200, 0), 3)

    return face_centers, frame


def show_frame(frame, window_name='Face Picker'):
    cv2.imshow(window_name, frame)
    k = cv2.waitKey(6) & 0xff
    return k


"""FACE DETECTION LOOP ___________________________________________________________________"""

try:
    while True:
        frame = vs.read()
        face_positions, new_frame = find_faces_dnn(frame)
        show_frame(new_frame)

except KeyboardInterrupt:
    print("Closing camera connection")
    cv2.destroyAllWindows()
    vs.stop()

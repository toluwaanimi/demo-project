import sys
import cv2
import time
import numpy as np


class FaceDetector:
    def __init__(self, use_raspberry_pi_camera=False, video_resolution=(700, 400), frame_rate=13):
        self.use_raspberry_pi_camera = use_raspberry_pi_camera
        self.video_resolution = video_resolution
        self.frame_rate = frame_rate
        self.mid_point = (video_resolution[0] // 2, video_resolution[1] // 2)

        # Switching to a different pre-trained model for face detection
        self.model = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt",
                                              "models/face_recognition.caffemodel")

        self.video_stream = cv2.VideoCapture(0)
        self.video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, video_resolution[0])
        self.video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, video_resolution[1])
        self.video_stream.set(cv2.CAP_PROP_FPS, frame_rate)

        time.sleep(0.2)

    def get_face_centers(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        detections = self.model.forward()
        face_centers = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face_center = (int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2))
            face_centers.append(((face_center[0] - self.mid_point[0]), (face_center[1] - self.mid_point[1])))

        return face_centers, frame

    def display_frame(self, frame, window_name='Face Detector'):
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(6) & 0xff
        return key

    def process_frame(self, frame):
        face_centers, frame = self.get_face_centers(frame)
        for center in face_centers:
            cv2.circle(frame, center, 4, (0, 200, 0), 3)
        return frame

    def run(self):
        try:
            while True:
                ret, frame = self.video_stream.read()
                if ret:
                    frame = self.process_frame(frame)
                    self.display_frame(frame)
        except KeyboardInterrupt:
            print("Closing camera connection")
            cv2.destroyAllWindows()
            self.video_stream.release()


if __name__ == "__main__":
    detector = FaceDetector()
    detector.run()

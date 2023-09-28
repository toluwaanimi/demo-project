import cv2


def calculate_movement(center, frame_center):
    dx = center[0] - frame_center[0]
    dy = center[1] - frame_center[1]
    return dx, dy


# Initialize the Haar cascade detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_center = (frame_width // 2, frame_height // 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_center = (x + w // 2, y + h // 2)
        cv2.circle(frame, face_center, 5, (0, 255, 0), -1)
        dx, dy = calculate_movement(face_center, frame_center)
        print(f"Face - X offset: {dx}, Y offset: {dy}")

    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        upper_body_center = (x + w // 2, y + h // 2)
        cv2.circle(frame, upper_body_center, 5, (255, 0, 255), -1)
        dx, dy = calculate_movement(upper_body_center, frame_center)
        print(f"Upper Body - X offset: {dx}, Y offset: {dy}")

    cv2.imshow("Face and Upper Body Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

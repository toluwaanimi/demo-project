
import URBasic
import math
import numpy as np
import sys
import cv2
import time
import imutils
from imutils.video import VideoStream
import math3d as m3d
import serial
import argparse
import threading

"""SETTINGS AND VARIABLES ________________________________________________________________"""

RASPBERRY_BOOL = False
# If this is run on a linux system, a picamera will be used.
# If you are using a linux system, with a webcam instead of a raspberry pi delete the following if-statement
if sys.platform == "linux":
    import picamera
    from picamera.array import PiRGBArray

    RASPBERRY_BOOL = True

ROBOT_IP = '192.168.0.191'
ACCELERATION = 0.9  # Robot acceleration value
VELOCITY = 0.8  # Robot speed value

# The Joint position the robot starts at
robot_starting_position = (math.radians(-218),
                           math.radians(-63),
                           math.radians(-93),
                           math.radians(-20),
                           math.radians(88),
                           math.radians(0))

# Path to the face-detection model:
pretrained_model = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt",
                                            "models/face_recognition.caffemodel")

video_resolution = (700, 400)  # resolution the video capture will be resized to, smaller sizes can speed up detection
video_midpoint = (int(video_resolution[0] / 2),
                  int(video_resolution[1] / 2))
video_asp_ratio = video_resolution[0] / video_resolution[1]  # Aspect ration of each frame
video_viewangle_hor = math.radians(25)  # Camera FOV (field of fiew) angle in radians in horizontal direction

# Variable which scales the robot movement from pixels to meters.
m_per_pixel = 00.00009

# Size of the robot view-window
# The robot will at most move this distance in each direction
max_x = 0.2
max_y = 0.2

# Maximum Rotation of the robot at the edge of the view window
hor_rot_max = math.radians(50)
vert_rot_max = math.radians(25)

vs = VideoStream(src=0,
                 usePiCamera=RASPBERRY_BOOL,
                 resolution=video_resolution,
                 framerate=13,
                 meter_mode="backlit",
                 exposure_mode="auto",
                 shutter_speed=8900,
                 exposure_compensation=2,
                 rotation=0).start()
time.sleep(0.2)

"""FUNCTIONS _____________________________________________________________________________"""


def find_faces_dnn(image):
    """
    Finds human faces in the frame captured by the camera and returns the positions
    uses the pretrained model located at pretrained_model

    Input:
        image: frame captured by the camera

    Return Values:
        face_centers: list of center positions of all detected faces
            list of lists with 2 values (x and y)
        frame: new frame resized with boxes and probabilities drawn around all faces

    """

    frame = image
    frame = imutils.resize(frame, width=video_resolution[0])

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    pretrained_model.setInput(blob)

    # the following line handles the actual face detection
    # it is the most computationally intensive part of the entire program
    detections = pretrained_model.forward()
    face_centers = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.4:
            continue

        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated probability
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


def check_max_xy(xy_coord):
    """
    Checks if the face is outside of the predefined maximum values on the lookaraound plane

    Inputs:
        xy_coord: list of 2 values: x and y value of the face in the lookaround plane.
            These values will be evaluated against max_x and max_y

    Return Value:
        x_y: new x and y values
            if the values were within the maximum values (max_x and max_y) these are the same as the input.
            if one or both of the input values were over the maximum, the maximum will be returned instead
    """

    x_y = [0, 0]

    if -max_x <= xy_coord[0] <= max_x:
        # checks if the resulting position would be outside of max_x
        x_y[0] = xy_coord[0]
    elif -max_x > xy_coord[0]:
        x_y[0] = -max_x
    elif max_x < xy_coord[0]:
        x_y[0] = max_x
    else:
        raise Exception(" x is wrong somehow:", xy_coord[0], -max_x, max_x)

    if -max_y <= xy_coord[1] <= max_y:
        # checks if the resulting position would be outside of max_y
        x_y[1] = xy_coord[1]
    elif -max_y > xy_coord[1]:
        x_y[1] = -max_y
    elif max_y < xy_coord[1]:
        x_y[1] = max_y
    else:
        raise Exception(" y is wrong somehow", xy_coord[1], max_y)

    return x_y


def set_face_origin(robot):
    """
    Creates a new coordinate system at the current robot tcp position.
    This coordinate system is the basis of the face following.
    It describes the midpoint of the plane in which the robot follows faces.

    Args:
        robot (Robot): The robot object.

    Returns:
        Transform: The new coordinate system.
    """
    position = robot.get_actual_tcp_pose()
    return m3d.Transform(position)


def move_to_face(list_of_facepos, robot_pos):
    """
    Function that moves the robot to the position of the face

    Inputs:
        list_of_facepos: a list of face positions captured by the camera, only the first face will be used
        robot_pos: position of the robot in 2D - coordinates

    Return Value:
        prev_robot_pos: 2D robot position the robot will move to. The basis for the next call to this funtion as robot_pos
    """

    face_from_center = list(list_of_facepos[0])

    prev_robot_pos = robot_pos
    scaled_face_pos = [c * m_per_pixel for c in face_from_center]

    robot_target_xy = [a + b for a, b in zip(prev_robot_pos, scaled_face_pos)]
    # print("..", robot_target_xy)

    robot_target_xy = check_max_xy(robot_target_xy)
    prev_robot_pos = robot_target_xy

    x = robot_target_xy[0]
    y = robot_target_xy[1]
    z = 0
    xyz_coords = m3d.Vector(x, y, z)

    x_pos_perc = x / max_x
    y_pos_perc = y / max_y

    x_rot = x_pos_perc * hor_rot_max
    y_rot = y_pos_perc * vert_rot_max * -1

    tcp_rotation_rpy = [y_rot, x_rot, 0]
    tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
    position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)

    oriented_xyz = origin * position_vec_coords
    oriented_xyz_coord = oriented_xyz.get_pose_vector()

    coordinates = oriented_xyz_coord

    qnear = robot.get_actual_joint_positions()
    next_pose = coordinates
    robot.set_realtime_pose(next_pose)

    return prev_robot_pos



def run_face_tracking():
    """FACE TRACKING LOOP ____________________________________________________________________"""

    # Initialise robot with URBasic
    print("initialising robot")
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

    robot.reset_error()
    print("robot initialised")
    time.sleep(1)

    # Move Robot to the midpoint of the lookplane
    robot.movej(q=robot_starting_position, a=ACCELERATION, v=VELOCITY)

    robot_position = [0, 0]
    origin = set_face_origin(robot)

    robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
    time.sleep(1)  # just a short wait to make sure everything is initialised

    try:
        print("starting face tracking loop")
        while True:

            frame = vs.read()
            face_positions, new_frame = find_faces_dnn(frame)
            show_frame(new_frame)
            if len(face_positions) > 0:
                robot_position = move_to_face(face_positions, robot_position)

        print("exiting face tracking loop")
    except KeyboardInterrupt:
        print("closing robot connection")
        # Remember to always close the robot connection, otherwise it is not possible to reconnect
        robot.close()
    except:
        robot.close()
        


def run_heatbeat_listener():
        # Define the serial port and baud rate
    serial_port = '/dev/ttyACM0'  # This may vary depending on your Arduino model
    # baud_rate = 9600
    baud_rate  = 115200

    # Initialize the serial connection
    ser = serial.Serial(serial_port, baud_rate)

    try:
        while True:
            data = ser.readline().decode('utf-8').strip()  # Read data from Arduino
            print(data)  # Print the received data
    except KeyboardInterrupt:
        ser.close()  # Close the serial connection when Ctrl+C is pressed
    


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Face Tracking and Other Program')
parser.add_argument('--face-tracking', action='store_true', help='Enable face tracking')
parser.add_argument('--heartbeat', action='store_true', help='Run HeartBeat Program')
parser.add_argument('--both', action='store_true', help='Run both face tracking and HeartBeat')
args = parser.parse_args()

if args.both:
    # Run both face tracking and other program together
    face_tracking_thread = threading.Thread(target=run_face_tracking)
    face_tracking_thread.start()
    run_heatbeat_listener()
    
elif args.face_tracking:
    face_tracking_thread = threading.Thread(target=run_face_tracking)
    face_tracking_thread.start()

elif args.heartbeat:
    run_heatbeat_listener()
else:
    # Prompt user to choose which program to run
    print("Please select an option:")
    print("1. Run Face Tracking")
    print("2. Run HeartBeat Program")
    print("3. Run Full Application")
    choice = input("Enter the number of your choice: ")

    if choice == '1':
        run_face_tracking()
    elif choice == '2':
        run_heatbeat_listener()
    elif choice == "3":
        run_face_tracking()
        run_heatbeat_listener()
    else:
        print("Invalid choice. Please enter 1 or 2.")
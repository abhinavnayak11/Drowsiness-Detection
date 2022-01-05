import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import os

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(1)

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	EAR = (A + B) / (2.0 * C)
	return EAR

flag = 0

main_loop = 0

while True:
    _, frame = cap.read()
    frame_orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)

    EARS = np.array([])
    for face in faces:

        # if main_loop == 0:
        #     print(type(face))
        #     face_img = frame[face.top():face.bottom(), face.left():face.right()]
        #     face_img = cv2.resize(face_img, (75,75))
        #     print(face_img.shape)
        #     main_loop = 1

        landmarks = landmark_detector(gray, face)
        left_eye_points = []
        right_eye_points = []
        for i in range(36,42):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            left_eye_points.append((x, y))
        
        for i in range(42, 48):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            right_eye_points.append((x, y))
        
        left_eye_points = np.array(left_eye_points)
        right_eye_points = np.array(right_eye_points)

        leftEAR = eye_aspect_ratio(left_eye_points)
        rightEAR = eye_aspect_ratio(right_eye_points)
        EAR = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(left_eye_points)
        rightEyeHull = cv2.convexHull(right_eye_points)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        EARS = np.append(EARS, [EAR])

    if faces:   # if no faces detected, EARS variable will be empty. Thus this will avoid error.

        EAR_avg = EARS.min()        # select the minimum EAR value. 
        face_idx = EARS.argmin()    # select the face index with min EAR value
        face = faces[face_idx]      # select the face with min EAR value

        if EAR_avg < 0.22:          # display the face with EAR value < 0.22
            flag += 1
            if flag >= 10:          # only alert if drowsy for more than 10 frames
                cv2.putText(frame, "****************ALERT!****************", (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (100,450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                face_img = frame_orig[face.top():face.bottom(), face.left():face.right()]
                face_img = cv2.resize(face_img, (100,100))

                frame[45:145, 270:370] = face_img
                print('\a', end='')     # to make sound
        else:
            flag = 0
    
    cv2.imshow('drowsiness detector', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
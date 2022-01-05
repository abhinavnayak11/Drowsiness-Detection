'''
* Copyright (c) 2017 Akshay Bahadur. All rights reserved.
* Copyrights licensed under the MIT License
* License Terms at the bottom.
'''

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import os

from src.alert import alert_drowsy_person

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(1)

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	EAR = (A + B) / (2.0 * C)
	return EAR

def calc_EAR(frame, face):

    landmarks = landmark_detector(frame, face)
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

    return frame, EAR

flag = 0

if __name__=="__main__":

    while True:
        _, frame = cap.read()
        frame_orig = frame.copy()       # this will be used to extract face (w/o eye contours) of drowsy person
        faces = face_detector(frame)

        EARS = np.array([])
        for face in faces:

            frame, EAR = calc_EAR(frame, face)

            EARS = np.append(EARS, [EAR])

        frame, flag = alert_drowsy_person(frame_orig, frame, faces, EARS, flag)
        
        cv2.imshow('drowsiness detector', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

'''
MIT License

Copyright (c) 2017 Akshay Bahadur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
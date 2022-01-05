import cv2

def alert_drowsy_person(frame_orig, frame, faces, EARS, flag):
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
                print('\a')     # to make sound
        else:
            flag = 0
        
    return frame, flag
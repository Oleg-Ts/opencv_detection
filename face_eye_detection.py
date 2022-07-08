import cv2
import numpy as np


### Face and eye detection

def face_eye_detection():
    '''
    The control keys are:
    1. Only face detection - press 'f'
    2. Only eye detection - press 'e'
    3. Both face and eye detection - press 'b'
    4. Both face and eye detection + face inversion - press 'i'
    5. Original - press other key
    '''

    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    scaling_factor = 1.0

    mode = None
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        c = cv2.waitKey(1)
        if c == 27:
            break

        if c != -1 and c != 255 and c != mode:
            mode = c

        if mode in (ord('f'), ord('e'), ord('b'), ord('i')):
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3)

            for (x,y,w,h) in face_rects:
                eyes = eye_cascade.detectMultiScale(gray_frame[y:y+h, x:x+w], scaleFactor=1.3, minNeighbors=7)
                for (x_eye,y_eye,w_eye,h_eye) in eyes:
                    center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
                    radius = int(0.3 * (w_eye + h_eye))
                if mode in (ord('f'), ord('b'), ord('i')):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color = (255,0,255), thickness = 2)
                if mode in (ord('e'), ord('b'), ord('i')):
                    cv2.circle(frame[y:y+h, x:x+w], center, radius, color = (255, 0, 0), thickness = 1)
                if mode == ord('i'):
                    src_points = np.float32([[0, 0], [0, w-1], [h-1, 0]])
                    dst_points = np.float32([[h-1, w-1], [h-1, 0], [0, w-1]])
                    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
                    face_frame = frame[y:y+h, x:x+w]
                    face_frame_mirror = cv2.warpAffine(frame[y:y+h, x:x+w], affine_matrix, (w, h))
                    frame[y:y+h, x:x+w] = face_frame_mirror 
        cv2.imshow('Face and eye detection', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_eye_detection()
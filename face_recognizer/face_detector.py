import numpy as np
import cv2


_DEFAULT_HAAR_NAME = 'haarcascade_frontalface_default.xml'

def get_faces(
        img: np.array, 
        haar_name: str=_DEFAULT_HAAR_NAME
        ) -> np.array:
    
    if isinstance(img, str):
        img = cv2.imread(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_name)
    faces_rect = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=9)

    faces = []
    rect_img = img.copy()

    for i, (x, y, w, h) in enumerate(faces_rect):
        cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = img[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA)
        faces.append(resized_face)

    np_faces = np.array(faces).astype(np.float32)
    np_faces /= 255

    return np_faces, faces, rect_img, faces_rect

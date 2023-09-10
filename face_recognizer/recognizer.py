import cv2
from datetime import datetime

from .face_detector import get_faces
from .face_recognizer import remember, get_top_n, recognize


def handle_image(img_filename: str):
    img = cv2.imread(img_filename)
    faces, cv2_faces, rect_img, faces_rect = get_faces(img)
    
    if len(faces):
        names = recognize(faces)

    # writing names onto image
    for i, ((x, y, w, h), val_names) in enumerate(zip(faces_rect, names)):
        if len(val_names):
            val, name = val_names[0]
            val = round(val, 4)
        else:
            val, name = '', 'NOT_REC'
        
        text_y_pos = y + 30
        cv2.putText(
                rect_img, 
                text=f'{i + 1}-{name}',
                org=(x, text_y_pos),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, 
                color=(100, 0, 0),
                thickness=2, 
                lineType=cv2.LINE_AA,
                )
        
        cv2.putText(
                rect_img, 
                text=f'{val}',
                org=(x, text_y_pos + 20),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.8, 
                color=(100, 0, 100),
                thickness=2, 
                lineType=cv2.LINE_AA,
                )
    
    fn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cv2.imwrite(f'data/recognized/{fn}.png', rect_img)

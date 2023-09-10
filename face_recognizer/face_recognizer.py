import torch
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

from .vae.vae import VAE
from .face_detector import get_faces


class FaceRecognizer:
    def __init__(self, weights_path, hidden_size, logger=None):
        self.__logger = logger if logger else logging.Logger(__name__)

        self.__logger.info('model load started')
        self.__vectors = []
        self.__vec2name = {}
        self.hidden_size = hidden_size

        self.model = VAE(hidden_size=hidden_size)

        self.state_dict = torch.load(
            weights_path, 
            map_location=torch.device('cpu')
            )
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        self.__logger.info('model loaded')

    def remember(self, faces: np.array, name: str):
        '''
        remember only first face in faces
        '''
        batch = torch.from_numpy(faces)
        mu, logsigma = self.model.encoder(batch)

        if len(mu.shape) != 2:
            return False # face not detected
        
        vec = tuple(mu[0].tolist())
        self.__vec2name[vec] = name
        self.__vectors.append(vec)
        return True

    def get_top_n(self, vec, n=5):
        cos_similarities = cosine_similarity(self.__vectors, [vec]).tolist()

        res = [
            (
                cos_similarities[i][0], 
                self.__vec2name[self.__vectors[i]]
            ) 
            for i in range(len(self.__vectors))
        ]

        l = min(n, len(res))
        return sorted(res, key=lambda x: x[0], reverse=True)[:l]

    def recognize(
            self,
            faces: np.array, 
            top_n=2, 
            threshold=0.5) -> list[list[str]]:
        
        if not len(faces):
            raise Exception('len(faces) = 0')

        if not len(self.__vectors):
            return [] # none known face

        batch = torch.from_numpy(faces)
        mu, logsigma = self.model.encoder(batch)

        if len(mu.shape) != 2:
            return [] # no face detected
        
        mu_list = np.array(mu.tolist())
        res = [self.get_top_n(m) for m in mu_list]
        return res
    
    def remember_many(self, people, path=''):
        count = 0

        for name in people:
            for filename in people[name]:
                img = cv2.imread(f'{path}{filename}')
                faces, _, _, _ = get_faces(img)

                if len(faces):
                    res = self.remember(faces, name)
                    count += int(res)
        
        return count
    
    def recognize_from_image(self, img_path, save_path=None):
        img = cv2.imread(img_path)
        faces, _, rect_img, faces_rect = get_faces(img)

        names = []
        
        if len(faces):
            names = self.recognize(faces)

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
        
        if save_path:
            cv2.imwrite(save_path, rect_img)

        return rect_img

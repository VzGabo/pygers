import cv2
import torch
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1

class FaceRecognitionModel:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            select_largest = True,
            keep_all = True,
            min_face_size = 10,
            thresholds = [0.6, 0.7, 0.7],
            post_process = False,
            image_size = 160,
            device = self.device
        )
        self.encoder = InceptionResnetV1(pretrained='vggface2', classify=False, device=self.device).eval()

    def detect_faces(self, image):
        return self.mtcnn.detect(image, landmarks=True)

    def get_embeddings(self, face):
        return self.encoder.forward(face.reshape((1,3, 160, 160))).detach().cpu()

    def compare_faces(self, image1, image2):
        boxes, probs, landmarks = self.detect_faces(image1)
        boxes2, probs2, landmarks2 = self.detect_faces(image2)

        cara1 = self.mtcnn(image1)
        cara2 = self.mtcnn(image2)
        embedding_cara = self.get_embeddings(cara1)
        embedding_cara2 = self.get_embeddings(cara2)

        dist = torch.dist(embedding_cara, embedding_cara2, 2)
        return dist, boxes, landmarks, boxes2, landmarks2

    def process_image(self, image_path, image_path2):
        image = Image.open(image_path)
        image = image.convert('RGB')

        image2 = Image.open(image_path2)
        image2 = image2.convert('RGB')

        dist, boxes, landmarks, boxes2, landmarks2 = self.compare_faces(image, image2)
        print(dist)
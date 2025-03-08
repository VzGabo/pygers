
import torch
from PIL import Image
from typing import Dict, List, Tuple
import numpy as np
from io import BytesIO
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import euclidean

class FaceRecognitionModel:
    def __init__(self):
        self.known_face_embeddings: Dict[str, torch.Tensor] = {}
        self.umbral: float = 0.36
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.mtcnn = MTCNN(
            select_largest=False,
            keep_all=True,
            min_face_size=10,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,
            image_size=160,
            device=self.device
        ).eval()
        
        self.encoder = InceptionResnetV1(
            pretrained='vggface2',
            classify=False,
            device=self.device
        ).eval()

    def get_embeddings(self, faces: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(faces).detach().cpu()
        embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
        return embeddings

    def compare_face_with_known_faces(self, image: Image.Image) -> Tuple[List[Tuple[str, float]], List[np.ndarray]]:
        matches = []
        matched_boxes = []
        
        if not self.known_face_embeddings:
            return matches, matched_boxes
        
        processed_img = self.pre_process_image(image)
        
        boxes, probs, _ = self.mtcnn.detect(processed_img, landmarks=True)
        if boxes is None:
            return matches, matched_boxes
        
        faces = self.mtcnn.extract(processed_img, boxes, save_path=None)
        if faces is None:
            return matches, matched_boxes

        for i, face in enumerate(faces):
            print(i)
            face_embedding = self.get_embeddings(face.unsqueeze(0))
            for path, known_embedding in self.known_face_embeddings.items():
                distance = euclidean(face_embedding.view(-1), known_embedding.view(-1))
                print(f"{path} - {distance}")

                if distance <= self.umbral:
                    matches.append((path, distance))
                    matched_boxes.append(boxes[i])
        return self.post_process_image(matches, matched_boxes)
    

    def add_known_face(self, faces_path: List[str]) -> None:
        self.known_face_embeddings = {}

        for path in faces_path:
            img = Image.open(path)
            processed_img = self.pre_process_image(img)
            faces = self.mtcnn(processed_img)
            if faces is not None and len(faces) > 0:
                face = faces[0].unsqueeze(0)
                embedding = self.get_embeddings(face)
                self.known_face_embeddings[path] = embedding

    def scale_boxes(self, boxes: np.ndarray, original_size: Tuple[int, int], processed_size: Tuple[int, int]) -> np.ndarray:
        scale_x = original_size[0] / processed_size[0]
        scale_y = original_size[1] / processed_size[1]
        return boxes * np.array([scale_x, scale_y, scale_x, scale_y])

    def post_process_image(self, matches, matched_boxes):
        for i in range(len(matches)):
            img_path = matches[i][0]
            img_original_size = Image.open(img_path).size
            matched_boxes[i] = self.scale_boxes(matched_boxes[i], img_original_size, (400, 400))

        return matches, matched_boxes


    def pre_process_image(self, image: Image.Image) -> Image.Image:
        sizes = (400, 400)
        img = image.convert("RGB")
        exif = img.getexif()
        orientation = exif.get(274, 1)
        
        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
        
        return img.resize(sizes, Image.LANCZOS)

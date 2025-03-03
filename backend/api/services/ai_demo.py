import threading
import cv2
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
            face_embedding = self.get_embeddings(face.unsqueeze(0))
            for path, known_embedding in self.known_face_embeddings.items():
                distance = euclidean(face_embedding.view(-1), known_embedding.view(-1))
                if distance <= self.umbral:
                    matches.append((path, distance))
                    matched_boxes.append(boxes[i])
        
        return self.post_process_image(matches, matched_boxes)

    def add_known_face(self, faces_path: List[str]) -> None:
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

def main():
    model = FaceRecognitionModel()
    model.add_known_face(["abi3.png"])
    
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
    capture.set(cv2.CAP_PROP_FPS, 15)
    
    frame_counter = 0
    detection_interval = 5
    
    # Variables compartidas con protección de hilos
    lock = threading.Lock()
    latest_boxes = []
    latest_matches = []
    latest_similarities = []
    processing_thread = None
    
    def process_image(image: Image.Image, original_size: Tuple[int, int]):
        nonlocal latest_boxes, latest_matches, latest_similarities
        matches, boxes, similarities = model.compare_face_with_known_faces(image)
        
        # Escalar las coordenadas de los bounding boxes al tamaño original
        if boxes is not None and len(boxes) > 0:  # Verificar si hay boxes
            scale_x = original_size[0] / image.size[0]
            scale_y = original_size[1] / image.size[1]
            boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y])
        else:
            boxes = None  # Asegurarnos de que boxes sea None si no hay detecciones
        
        with lock:
            if boxes is not None and len(boxes) > 0:
                latest_boxes[:] = boxes
                latest_matches[:] = matches
                latest_similarities[:] = similarities
            else:
                latest_boxes.clear()
                latest_matches.clear()
            latest_similarities.clear()
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        if cv2.waitKey(1) == ord('q'):
            break
        
        # Convertir a PIL y manejar detección en hilo
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Redimensionar la imagen para el procesamiento
        processed_size = (400, 400)  # Tamaño al que se redimensiona la imagen en pre_process_image
        resized_img = pil_img.resize(processed_size, Image.LANCZOS)
        
        # Iniciar procesamiento en hilo si está listo
        if frame_counter % detection_interval == 0:
            if processing_thread is None or not processing_thread.is_alive():
                processing_thread = threading.Thread(
                    target=process_image, 
                    args=(resized_img, pil_img.size)  # Pasamos el tamaño original de la imagen
                )
                processing_thread.start()
        
        # Obtener copia segura de los datos
        with lock:
            current_boxes = list(latest_boxes)
            current_matches = list(latest_matches)
        
        # Dibujar resultados
        for i, box in enumerate(current_boxes):
            box = box.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            if i < len(current_matches):
                name, distance = current_matches[i]
                # Mostrar el nombre y la distancia (similitud) en el frame
                text = f"{name}: {distance:.2f}"
                cv2.putText(frame, text, (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
                
        cv2.imshow("Detección en Tiempo Real", frame)
        frame_counter += 1
    
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
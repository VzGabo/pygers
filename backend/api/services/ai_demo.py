import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import threading
import torch
import numpy as np
from typing import Dict, List, Tuple
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import euclidean

class FaceRecognitionModel:
    def __init__(self):
        self.known_face_embeddings: Dict[str, torch.Tensor] = {}
        self.umbral: float = 0.45
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

    def compare_face_with_known_faces(self, image: Image.Image) -> Tuple[List[Tuple[str, float]], List[np.ndarray], List[float]]:
        matches = []
        matched_boxes = []
        distances = []
        
        if not self.known_face_embeddings:
            return matches, matched_boxes, distances
        
        processed_img = self.pre_process_image(image)
        
        boxes, probs, _ = self.mtcnn.detect(processed_img, landmarks=True)
        if boxes is None:
            return matches, matched_boxes, distances
        
        faces = self.mtcnn.extract(processed_img, boxes, save_path=None)
        if faces is None:
            return matches, matched_boxes, distances

        for i, face in enumerate(faces):
            face_embedding = self.get_embeddings(face.unsqueeze(0))
            min_distance = float('inf')
            match_name = "Desconocido"
            for path, known_embedding in self.known_face_embeddings.items():
                distance = euclidean(face_embedding.view(-1), known_embedding.view(-1))
                if distance < min_distance:
                    min_distance = distance
                    match_name = path if distance <= self.umbral else "Desconocido"
            matches.append((match_name, min_distance))
            matched_boxes.append(boxes[i])
            distances.append(min_distance)
        
        return matches, matched_boxes, distances

    def add_known_face(self, faces_path: List[str]) -> None:
        for path in faces_path:
            img = Image.open(path)
            processed_img = self.pre_process_image(img)
            faces = self.mtcnn(processed_img)
            if faces is not None and len(faces) > 0:
                face = faces[0].unsqueeze(0)
                embedding = self.get_embeddings(face)
                self.known_face_embeddings[path] = embedding

    def pre_process_image(self, image: Image.Image) -> Image.Image:
        sizes = (800, 600)
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

<<<<<<< HEAD
class App:
    def __init__(self, window, window_title, model):
        self.window = window
        self.window.title(window_title)
        self.model = model
        self.capture = None
        self.is_running = False
        self.frame_counter = 0
        self.detection_interval = 5
        self.lock = threading.Lock()
        self.latest_boxes = []
        self.latest_matches = []
        self.latest_similarities = []
        self.processing_thread = None

        # UI elements
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.btn_load_image = tk.Button(window, text="Load Image", width=20, command=self.load_image)
        self.btn_load_image.pack(pady=5)

        self.btn_start = tk.Button(window, text="Start Camera", width=20, command=self.start_camera)
        self.btn_start.pack(pady=5)

        self.btn_stop = tk.Button(window, text="Stop Camera", width=20, command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop.pack(pady=5)

        self.delay = 15  # ms
        self.window.mainloop()

    def load_image(self):
        filename = filedialog.askopenfilename(initialdir=".", title="Select an image",
                                              filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")))
        if filename:
            self.model.add_known_face([filename])
            print(f"Loaded image: {filename}")

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error: Could not open video device.")
            return
        # Aumentar la resolución de la cámara
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.update()

    def stop_camera(self):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if self.capture:
            self.capture.release()

    def process_image(self, image: Image.Image):
        matches, boxes, similarities = self.model.compare_face_with_known_faces(image)
        with self.lock:
            self.latest_boxes[:] = boxes if boxes is not None else []
            self.latest_matches[:] = matches if matches is not None else []
            self.latest_similarities[:] = similarities if similarities is not None else []

    def update(self):
        if self.is_running:
            ret, frame = self.capture.read()
            if ret:
                # Convertir el frame a PIL.Image y aplicar preprocesamiento
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                processed_img = self.model.pre_process_image(pil_img)
=======
def main():
    model = FaceRecognitionModel()
    model.add_known_face(["backend/api/images/joshua.jpeg", "backend/api/images/me_presentation.png"])
    
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
            
>>>>>>> 7e6674e1d7895f289e933a6f6616769fa54c026f
                
                
                # Procesamiento en hilo cada 'detection_interval' frames
                if self.frame_counter % self.detection_interval == 0:
                    if self.processing_thread is None or not self.processing_thread.is_alive():
                        self.processing_thread = threading.Thread(
                            target=self.process_image, 
                            args=(processed_img,)
                        )
                        self.processing_thread.start()
                
                # Dibujar bounding boxes y texto en la imagen procesada
                draw = ImageDraw.Draw(processed_img)
                with self.lock:
                    current_boxes = list(self.latest_boxes)
                    current_matches = list(self.latest_matches)
                
                for i, box in enumerate(current_boxes):
                    box = box.astype(int)
                    draw.rectangle([box[0], box[1], box[2], box[3]], outline="green", width=2)
                    if i < len(current_matches):
                        name, distance = current_matches[i]
                        text = f"{name.split('/')[-1]}: {distance:.2f}"
                        draw.text((box[0], box[1] - 10), text, fill="green")
                
                # Mostrar la imagen procesada en el canvas
                self.photo = ImageTk.PhotoImage(image=processed_img)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.canvas.image = self.photo
                self.frame_counter += 1
            self.window.after(self.delay, self.update)

if __name__ == "__main__":
    model = FaceRecognitionModel()
    root = tk.Tk()
    app = App(root, "Face Recognition App", model)
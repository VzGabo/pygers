import cv2
import torch
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1

def capture_camera():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        select_largest=True,
        keep_all=True,
        min_face_size=10,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,
        image_size=160,
        device=device
    )

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("No se pudo abrir la cámara")
        return

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_FPS, 15)

    frame_counter = 0
    detection_interval = 5
    scale_factor = 0.5 

    while True:
        frame_exist, frame = capture.read()
        if not frame_exist:
            break

        if cv2.waitKey(1) == ord('q'):
            break

        if frame_counter % detection_interval == 0:
            # escalar el frame
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            boxes, probs, landmarks = mtcnn.detect(small_frame_rgb, landmarks=True)

            # Escalar las coordenadas al tamaño original
            if boxes is not None:
                boxes = boxes / scale_factor
                landmarks = landmarks / scale_factor
            else:
                boxes = None

        # Dibujar bounding boxes y landmarks
        if boxes is not None:
            for box, landmark in zip(boxes, landmarks):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color=(0, 255, 0),  # Color verde
                    thickness=2
                )

                # Dibujar landmarks
                for point in landmark:
                    x, y = map(int, point)
                    cv2.circle(
                        frame,
                        (x, y),
                        radius=2,
                        color=(0, 0, 255),  # Color rojo
                        thickness=-1  # Relleno
                    )

        # Mostrar el frame
        cv2.imshow("Detección de Rostros", frame)
        frame_counter += 1

    # Liberar recursos
    capture.release()
    cv2.destroyAllWindows()


def detect_image(image_path, image_path2):
    image = Image.open(image_path)
    image = image.convert('RGB')

    image2 = Image.open(image_path2)
    image2 = image2.convert('RGB')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        select_largest=True,
        keep_all=True,
        min_face_size= 10,
        thresholds= [0.6, 0.7, 0.7],
        post_process= False,
        image_size= 160,
        device= device
    )

    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    # boxes [x1, y1, x2, y2]
    # landmarks [[x1, y1], [x2, y2], ..., [xn, yn]] eyes, noses, mouths...
    boxes2, probs2, landmarks2 = mtcnn.detect(image2, landmarks=True)

    encoder = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()

    cara1 = mtcnn(image)
    cara2 = mtcnn(image2)
    embedding_cara = encoder.forward(cara1.reshape((1,3, 160, 160))).detach().cpu()
    embedding_cara2 = encoder.forward(cara2.reshape((1,3, 160, 160))).detach().cpu()

    dist = torch.dist(embedding_cara, embedding_cara2, 2)


    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.imshow(image)
    for box, landmarks in zip(boxes, landmarks):
        ax1.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c="red")
        rect = plt.Rectangle(
            xy = (box[0], box[1]),
            width = box[2] - box[0],
            height = box[3] - box[1],
            fill = False,
            edgecolor = 'red',
            linewidth = 2
        )

        ax1.add_patch(rect)

    ax1.axis('off')

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(image2)

    for box, landmarks in zip(boxes2, landmarks2):
        ax2.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c="red")
        rect = plt.Rectangle(
            xy = (box[0], box[1]),
            width = box[2] - box[0],
            height = box[3] - box[1],
            fill = False,
            edgecolor = 'red',
            linewidth = 2
        )

        ax2.add_patch(rect)

    ax2.axis('off')

    fig1.show()
    fig2.show()
    plt.show()



    

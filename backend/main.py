from typing import List, Annotated
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer
from PIL import Image
from fastapi.middleware import Middleware
import os
from io import BytesIO
from fastapi.responses import JSONResponse
from api.services.ai import FaceRecognitionModel

app = FastAPI()

'''
    Configuración para que conecte con el frontend
    a través de la clase Middleware
'''

origin = {
    "http://localhost:3000"
}

Middleware(
    origin = origin,
    allow_headers = ["*"],
    allow_credentials = True
)


# Initialize the model
face_model = FaceRecognitionModel()
oauth2 = OAuth2PasswordBearer(tokenUrl = "comparate")

UPLOAD_FOLDER = 'api/images'
os.makedirs(UPLOAD_FOLDER, exist_ok = True)

async def exist_images() -> bool:

    os.listdir()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/upload-images")
async def upload_image(images: List[UploadFile] = File(...)):
    try:
        # Clear existing images
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        list_path = []
        # Save new images
        for image in images:
            file_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image_data = await image.read()
            with open(file_path, "wb") as image_file:
                image_file.write(image_data)
            list_path.append(file_path)

        return JSONResponse(
            content={"message": "Images uploaded successfully"}, status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Comparate criminals with others suspects
# @app.post("/compare-face")
# async def upload_criminal(image: Annotated[UploadFile, "Image of the suspect"]):
MAX_SIZE = 10 * 1024 * 1024
@app.post("/compare-faces")
async def compare_faces(file: UploadFile = File(...)):

    """
    Este endpoint recibe una imagen y la compara contra las caras conocidas 
    almacenadas en face_model.known_face_embeddings.
    """
    # 1. Validar que sea una imagen
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado.")
    
    # 2. Guardar temporalmente el archivo o leerlo en memoria
    # Opción A: Leerlo directamente en memoria
    file_bytes = await file.read()
    if len(file_bytes) > MAX_SIZE:
        raise HTTPException(status_code = 400, detail = "El archivo excede de los parámetros")
    # 3. Crear un objeto PIL.Image para pasarlo al modelo
    try:
        image = Image.open(BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code = 400, detail = f"Error al abrir la imagen: {e}")
    


    
    # 4. Llamar al método compare_face_with_known_faces del modelo
    matches, matched_boxes = face_model.compare_face_with_known_faces(image)
    
    list_images = []
    
    for path_image in os.listdir(UPLOAD_FOLDER):
        with open(path_image, "rb") as path_img:
            list_images.append(path_img)
    # matches es una lista de tuplas (path, distance)
    # matched_boxes es la lista de bounding boxes correspondientes
    # 5. Procesar la respuesta para devolverla al frontend
    response_data = []
    for i, (path, distance) in enumerate(matches):

        response_data.append({
            "known_face_path": path,
            "distance": distance,
            "box": matched_boxes[i].tolist()  # Convertir a lista para JSON
        })
    
    if not response_data:
        return f"Array final: {response_data}"
    raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail = "Hay un error extraño")
    # if not response_data:
    #     return {"message": "No se encontraron coincidencias"}
    
    # return {
    #     "message": "Se encontraron coincidencias",
    #     "results": response_data
    # }

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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

'''
    Configuración para que conecte con el frontend
    a través de la clase Middleware
'''

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str

# Initialize the model
face_model = FaceRecognitionModel()
oauth2 = OAuth2PasswordBearer(tokenUrl = "comparate")

UPLOAD_FOLDER = 'api/images'
os.makedirs(UPLOAD_FOLDER, exist_ok = True)

'''
    FUNCTIONS WE'LL USE TO WORK IN THE COMPARATION
'''

async def validate_image(image: UploadFile):

    if not image.filename.lower().endswith(('.jpg', '.png', 'jpeg')):
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST,
                            detail = "Image format don't accept")
    file = await image.read()
    return file

'''
    ENDPOINTS WHERE WE'LL WORK THE CONNECTION FRONT AND BACK
'''

@app.get("/")
def root():
    return {"message": "Hola World"}

@app.get("/upload-faces")
async def get_images():
    if not os.listdir("api/images/"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail = "There is not images")
    return {"path_images" : os.listdir("api/images/")}
    
@app.get("/faces/{name}")
async def download_image(name):
    file_path = os.path.join(UPLOAD_FOLDER, name)
    print(file_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return FileResponse(file_path)


@app.post("/upload-faces")
async def upload_image(files: List[UploadFile] = File(...)):
    try:
        # Clear existing images
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        list_path = []
        # Save new images
        for image in files:
            file_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image_data = await image.read()
            with open(file_path, "wb") as image_file:
                image_file.write(image_data)
            list_path.append(file_path)

        face_model.add_known_face(list_path)

        return JSONResponse(
            content={"message": "Images uploaded successfully", "path" : list_path}, status_code = 200
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code = 500)

@app.post("/compare-faces")
async def compare_faces(file: UploadFile = File(...)):

    # Validar imagen
    if not file.filename.lower().endswith(('.jpg', '.png', 'jpeg')):
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST,
                            detail = "Image format don't accept")
    image_bytes = await file.read()

    # Conocer el resultado 
    try: 
        image_pil = Image.open(BytesIO(image_bytes))
    except Exception as e: 
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST,
                            detail = f"Err to open the image {e}")
    

    # Capturar las caras y comparar las imagenes 
    try:
        matches, matched_boxes = face_model.compare_face_with_known_faces(image_pil)
        response_data = []
        for i, (path, distance) in enumerate(matches):
            response_data.append({
                "known_face_path": path,
                "distance": distance,
                "box": matched_boxes[i].tolist()
            })
        if not response_data:
            return {"message": "No se encontraron coincidencias", "information" : response_data}
        return {"message": "Se encontraron coincidencias", "results": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from typing import List
from fastapi import FastAPI, File, UploadFile
import os
from fastapi.responses import JSONResponse
from api.services.ai import FaceRecognitionModel

app = FastAPI()

# Initialize the model
#face_model = FaceRecognitionModel()

UPLOAD_FOLDER = 'api/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/upload")
async def upload_image(images: List[UploadFile] = File(...)):
    try:
        # Clear existing images
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save new images
        for image in images:
            file_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image_data = await image.read()
            with open(file_path, "wb") as image_file:
                image_file.write(image_data)

        return JSONResponse(
            content={"message": "Images uploaded successfully"}, status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
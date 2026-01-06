import numpy as np
import keras
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from similarity_scorer import compare, scale_score
from PIL import Image


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

comparison_model = keras.models.load_model("comparison_model.keras")

@app.post("/predict")
async def scoring_endpoint(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    raw_score = compare(comparison_model, image)
    score = scale_score(raw_score)
    return {"score" : str(raw_score)}

@app.get("/win")
async def win_endpoint():
    daily_image = "./training_files/test_images/kyrie.jpg"
    return FileResponse(daily_image) 



import numpy as np
import keras
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from similarity_scorer import compare
from PIL import Image


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

comparison_model = keras.models.load_model("comparison_model.keras")

@app.post("/predict")
async def scoring_endpoint(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    score = compare(comparison_model, image)
    return {"score" : str(score)}
    



import keras
from fastapi import FastAPI
from similarity_scorer import compare

app = FastAPI()

comparison_model = keras.models.load_model("comparison_model.keras")

@app.get("/")
async def scoring_endpoint():
    image = ["./training_files/test_images/kyrie.jpg"]
    score = compare(comparison_model, image)
    return {"score" : str(score)}
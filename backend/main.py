import os, datetime, random
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from keras.applications import resnet
from fastapi.middleware.cors import CORSMiddleware

LEFT_DIR = "../data/left"
RIGHT_DIR = "../data/right"
MODEL_PATH = "./comparison_model.keras"
ARTIFACT_DIR = "../artifacts"
POOL_SIZE = 15
TARGET_SIZE = (200, 200)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = keras.models.load_model(MODEL_PATH)

# Serve dataset images so frontend can render them
app.mount("/images/left", StaticFiles(directory=LEFT_DIR), name="left_images")
app.mount("/images/right", StaticFiles(directory=RIGHT_DIR), name="right_images")

def load_ids_and_embs():
    ids_path = os.path.join(ARTIFACT_DIR, "ids.txt")
    embs_path = os.path.join(ARTIFACT_DIR, "right_embs.npy")
    if not (os.path.exists(ids_path) and os.path.exists(embs_path)):
        raise RuntimeError("Missing artifacts. Run scripts/precompute_right_embs.py first.")

    with open(ids_path, "r") as f:
        ids = [line.strip() for line in f.readlines() if line.strip()]
    embs = np.load(embs_path).astype(np.float32)  # (N,D)
    if embs.shape[0] != len(ids):
        raise RuntimeError("ids.txt length does not match right_embs.npy rows.")
    return ids, embs

IDS, RIGHT_EMBS = load_ids_and_embs()
ID_TO_INDEX = {id:i for i,id in enumerate(IDS)}

def pil_to_model_input(pil_img):
    pil_img = pil_img.convert("RGB").resize(TARGET_SIZE)
    x = np.array(pil_img, dtype=np.float32)  # 0..255
    x = np.expand_dims(x, axis=0)
    x = resnet.preprocess_input(x)
    return tf.convert_to_tensor(x, dtype=tf.float32)

@tf.function
def embed(batch_x):
    return model(batch_x, training=False)

def cosine_to_matrix(query_emb, matrix):
    # query_emb: (D,), matrix: (N,D) np.float32
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    M = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return M @ q  # (N,)

def pick_daily_id():
    today = datetime.date.today().isoformat()
    idx = hash(today) % len(IDS)
    return IDS[idx]

def build_daily_game():
    target_id = pick_daily_id()

    prompt_path = os.path.join(LEFT_DIR, f"{target_id}.jpg")
    prompt_img = Image.open(prompt_path)
    prompt_x = pil_to_model_input(prompt_img)
    prompt_emb = embed(prompt_x).numpy()[0]  # (D,)

    true_right_path = os.path.join(RIGHT_DIR, f"{target_id}.jpg")
    true_right_img = Image.open(true_right_path)
    true_right_x = pil_to_model_input(true_right_img)
    true_right_emb = embed(true_right_x).numpy()[0]  # (D,)

    sims = cosine_to_matrix(true_right_emb, RIGHT_EMBS)


    #sims = cosine_to_matrix(prompt_emb, RIGHT_EMBS)  # (N,)
    order = np.argsort(-sims)

    # pick hard negatives excluding the correct id
    candidates = []
    for idx in order:
        cid = IDS[idx]
        if cid == target_id:
            continue
        candidates.append(cid)
        if len(candidates) >= (POOL_SIZE - 1):
            break

    pool = candidates + [target_id]
    random.Random(target_id).shuffle(pool)  # deterministic shuffle for the day

    # optional: include similarity for debugging (remove later)
    pool_debug = [{"id": cid, "url": f"/images/right/{cid}.jpg", "sim": float(sims[ID_TO_INDEX[cid]])} for cid in pool]

    return {
        "game_id": f"{datetime.date.today().isoformat()}_{target_id}",
        "target_id": target_id,  # you can remove this once frontend works
        "prompt": {"id": target_id, "url": f"/images/left/{target_id}.jpg"},
        "candidates": [{"id": cid, "url": f"/images/right/{cid}.jpg"} for cid in pool],
        "pool_debug": pool_debug
    }

DAILY_GAME = build_daily_game()

@app.get("/game/today")
def game_today():
    print(RIGHT_EMBS.shape)
    return DAILY_GAME

@app.post("/game/guess")
def game_guess(payload: dict):
    guess_id = payload.get("guess_id")
    if not guess_id:
        return {"error": "guess_id required"}

    correct_id = DAILY_GAME["target_id"]
    return {
        "correct": guess_id == correct_id,
        "correct_id": correct_id,
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "pairs": len(IDS),
        "pool_size": POOL_SIZE,
        "artifact_dir": ARTIFACT_DIR,
    }


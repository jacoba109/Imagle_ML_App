import os
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications import resnet

TARGET_SIZE = (200, 200)

def pil_to_model_input(pil_img):
    pil_img = pil_img.convert("RGB").resize(TARGET_SIZE)
    x = np.array(pil_img, dtype=np.float32)          # 0..255
    x = np.expand_dims(x, axis=0)                    # (1, H, W, 3)
    x = resnet.preprocess_input(x)                   # match Pipeline A
    return tf.convert_to_tensor(x, dtype=tf.float32)

@tf.function
def embed(model, batch_x):
    return model(batch_x, training=False)

def cosine_sim_matrix(query_emb, emb_matrix):
    # query_emb: (D,)
    # emb_matrix: (N, D)
    q = tf.math.l2_normalize(query_emb, axis=-1)
    M = tf.math.l2_normalize(emb_matrix, axis=-1)
    return tf.linalg.matvec(M, q)  # (N,) cosine

def list_ids(left_dir, right_dir):
    left_files = {os.path.splitext(f)[0] for f in os.listdir(left_dir) if f.lower().endswith((".jpg",".jpeg",".png"))}
    right_files = {os.path.splitext(f)[0] for f in os.listdir(right_dir) if f.lower().endswith((".jpg",".jpeg",".png"))}
    ids = sorted(left_files.intersection(right_files))
    return ids

def pick_daily_id(ids):
    # deterministic per day
    today = datetime.date.today().isoformat()
    idx = hash(today) % len(ids)
    return ids[idx]

def load_image(path):
    return Image.open(path)

def build_embeddings(model, image_paths, batch_size=64):
    embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch = tf.concat([pil_to_model_input(load_image(p)) for p in batch_paths], axis=0)
        e = embed(model, batch)  # (B, D)
        embs.append(e)
    return tf.concat(embs, axis=0)  # (N, D)

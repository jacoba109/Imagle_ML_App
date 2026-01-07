import os
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from keras.applications import resnet

RIGHT_DIR = "./data/right"
MODEL_PATH = "./backend/comparison_model.keras"
OUT_DIR = "./artifacts"
TARGET_SIZE = (200, 200)
BATCH_SIZE = 64

def pil_to_model_input(pil_img):
    pil_img = pil_img.convert("RGB").resize(TARGET_SIZE)
    x = np.array(pil_img, dtype=np.float32)   # 0..255
    x = np.expand_dims(x, axis=0)             # (1,H,W,3)
    x = resnet.preprocess_input(x)
    return tf.convert_to_tensor(x, dtype=tf.float32)

@tf.function
def embed(model, batch_x):
    return model(batch_x, training=False)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = keras.models.load_model(MODEL_PATH)

    # ids based on filenames like 0000.jpg
    ids = sorted([os.path.splitext(f)[0] for f in os.listdir(RIGHT_DIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    paths = [os.path.join(RIGHT_DIR, f"{id}.jpg") for id in ids]

    embs = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i+BATCH_SIZE]
        batch = tf.concat([pil_to_model_input(Image.open(p)) for p in batch_paths], axis=0)
        e = embed(model, batch)  # (B,D)
        embs.append(e.numpy())
        print(f"Embedded {min(i+BATCH_SIZE, len(paths))}/{len(paths)}")

    embs = np.concatenate(embs, axis=0).astype(np.float32)

    np.save(os.path.join(OUT_DIR, "right_embs.npy"), embs)
    with open(os.path.join(OUT_DIR, "ids.txt"), "w") as f:
        f.write("\n".join(ids))

    print("Saved:", embs.shape)

if __name__ == "__main__":
    main()

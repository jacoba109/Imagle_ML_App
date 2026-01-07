import tensorflow as tf
import numpy as np
import keras
import os
from keras.applications import resnet

target_shape = (200, 200)

def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

comparison_model = keras.models.load_model("./comparison_model.keras")

# Build paired lists (left[i] matches right[i])
left = sorted([f"../data/left/{f}" for f in os.listdir("../data/left")])
right = sorted([f"../data/right/{f}" for f in os.listdir("../data/right")])
n = min(len(left), len(right))
left, right = left[:n], right[:n]

# Shuffle pairs together (keeps alignment)
perm = np.random.RandomState(17).permutation(n)
left = [left[i] for i in perm]
right = [right[i] for i in perm]

# Deterministic "wrong right" negatives
neg_perm = np.random.RandomState(34).permutation(n)
neg = [right[i] for i in neg_perm]
for i in range(n):
    if neg[i] == right[i]:
        j = (i + 1) % n
        neg[i], neg[j] = neg[j], neg[i]

# Build dataset
ds = tf.data.Dataset.from_tensor_slices((left, right, neg))
ds = ds.map(lambda a,p,ne: (preprocess_image(a), preprocess_image(p), preprocess_image(ne)),
            num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)

def embed(x):
    return comparison_model(x)

def embed_resnet_preproc(x):
    return comparison_model(resnet.preprocess_input(x * 255.0))

cos = tf.keras.losses.CosineSimilarity(axis=1)

def cosine(a,b,eps=1e-8):
    a = tf.math.l2_normalize(a, axis=1)
    b = tf.math.l2_normalize(b, axis=1)
    return tf.reduce_sum(a*b, axis=1)

def run_benchmark(embed_fn, label):
    pos_sims = []
    neg_sims = []
    margins = []
    norms = []
    print("running first pipeline")
    i = 0
    for a,p,nimg in ds.take(50):
        print(f"taking! {i}")
        ea = embed_fn(a)
        ep = embed_fn(p)
        en = embed_fn(nimg)

        ps = cosine(ea, ep)
        ns = cosine(ea, en)

        pos_sims.append(ps.numpy())
        neg_sims.append(ns.numpy())
        margins.append((ps-ns).numpy())
        norms.append(tf.norm(ea, axis=1).numpy())
        i += 1

    pos_sims = np.concatenate(pos_sims)
    neg_sims = np.concatenate(neg_sims)
    margins  = np.concatenate(margins)
    norms    = np.concatenate(norms)

    print(f"\n=== {label} ===")
    print(f"pos cosine: mean={pos_sims.mean():.4f} std={pos_sims.std():.4f} p5={np.percentile(pos_sims,5):.4f} p95={np.percentile(pos_sims,95):.4f}")
    print(f"neg cosine: mean={neg_sims.mean():.4f} std={neg_sims.std():.4f} p5={np.percentile(neg_sims,5):.4f} p95={np.percentile(neg_sims,95):.4f}")
    print(f"margin (pos-neg): mean={margins.mean():.4f} std={margins.std():.4f} p5={np.percentile(margins,5):.4f} p95={np.percentile(margins,95):.4f}")
    print(f"ranking acc (pos>neg): {(margins>0).mean():.3f}")
    print(f"embed norm: mean={norms.mean():.4f} std={norms.std():.4f}")

run_benchmark(embed_resnet_preproc, "Pipeline A: x*255 then resnet.preprocess_input")
run_benchmark(embed, "Pipeline B: raw 0..1 resized (no resnet preprocess)")

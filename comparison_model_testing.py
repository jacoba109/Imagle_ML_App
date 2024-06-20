import tensorflow as tf
import numpy as np
import keras
import os
from keras import metrics
from keras.api.applications import resnet

target_shape = (200, 200)

def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

def preprocess_triplets(anchor, positive, negative):
    return (
        preprocess_image(anchor), 
        preprocess_image(positive), 
        preprocess_image(negative)
    )

comparison_model = keras.models.load_model("comparison_model.keras")

anchor_images = sorted([str("./training_files/left/" + f) for f in os.listdir("training_files/left")])
positive_images = sorted([str("./training_files/right/" + f) for f in os.listdir("training_files/right")])
image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

rng = np.random.RandomState(seed=17)
rng.shuffle(anchor_images)
rng.shuffle(positive_images)

negative_images = anchor_images + positive_images
np.random.RandomState(seed=34).shuffle(negative_images)

negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


for _ in range(5):
    sample = next(iter(val_dataset))
    anchor, positive, negative = sample

    anchor_embedding = comparison_model(resnet.preprocess_input(anchor))
    positive_embedding = comparison_model(resnet.preprocess_input(positive))
    negative_embedding = comparison_model(resnet.preprocess_input(negative))

    cosine_similarity = metrics.CosineSimilarity()
    positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
    negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
    pos_val = positive_similarity.numpy() * 1000 % 10
    neg_val = negative_similarity.numpy() * 1000 % 10

    print(f"Raw: {positive_similarity}, {negative_similarity} | Positive: {pos_val}, Negative: {neg_val}")

"""
test_anchor = ["./training_files/test_images/white.jpg"]
test_positive = ["./training_files/test_images/pretzel.jpg"]
test_negative = ["./training_files/test_images/black.jpg"]

test_a_ds = tf.data.Dataset.from_tensor_slices(test_anchor).map(preprocess_image).batch(1)
test_p_ds = tf.data.Dataset.from_tensor_slices(test_positive).map(preprocess_image).batch(1)
test_n_ds = tf.data.Dataset.from_tensor_slices(test_negative).map(preprocess_image).batch(1)

anchor = next(iter(test_a_ds))
positive = next(iter(test_p_ds))
negative = next(iter(test_n_ds))


anchor_embedding = comparison_model(resnet.preprocess_input(anchor))
positive_embedding = comparison_model(resnet.preprocess_input(positive))
negative_embedding = comparison_model(resnet.preprocess_input(negative))

cosine_similarity = metrics.CosineSimilarity()
positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
pos_val = positive_similarity.numpy()
neg_val = negative_similarity.numpy()

print(f"Positive: {pos_val}, Negative: {neg_val}")
"""
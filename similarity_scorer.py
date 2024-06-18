import tensorflow as tf
import numpy as np
from keras import metrics
from keras.api.applications import resnet
from PIL import Image

def compare(model, image):

    target_shape = (200, 200)
    anchor = ["./training_files/test_images/pretzel.jpg"]
    
    def preprocess_anchor(filename):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_shape)
        return image
    
    def preprocess_image(image: Image.Image) -> np.ndarray:
        image = image.resize((200, 200))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    anchor_ds = tf.data.Dataset.from_tensor_slices(anchor).map(preprocess_anchor).batch(1)
    anchor_tensor = next(iter(anchor_ds))
    
    image_tensor = preprocess_image(image)

    anchor_embedding = model(resnet.preprocess_input(anchor_tensor))
    image_embedding = model(resnet.preprocess_input(image_tensor))

    cosine_similarity = metrics.CosineSimilarity()
    similarity = cosine_similarity(anchor_embedding, image_embedding)

    return similarity.numpy()





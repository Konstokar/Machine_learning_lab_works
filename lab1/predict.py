import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = (64, 64)

model = tf.keras.models.load_model("ffnn_model.keras")

# загружаем реальные классы
with open("class_names.json", "r") as f:
    class_names = json.load(f)


def predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)

    print("Prediction:", class_names[class_id])


predict("test.jpg")

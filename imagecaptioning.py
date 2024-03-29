import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = InceptionV3(weights='imagenet')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def generate_caption(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    caption = decode_predictions(predictions, top=1)[0][0][1]
    return caption

img_path = '/content/drive/MyDrive/Deep Learning/Module 1 - Dataset/cars.jpg'
caption = generate_caption(img_path)
print("Predicted caption:", caption)

img = image.load_img(img_path)
plt.imshow(img)
plt.axis('off')
plt.show()


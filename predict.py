import tensorflow as tf
from PIL import Image
import numpy as np

# Load the custom SSD model
model = tf.saved_model.load('./workspace/training_demo/exported-model/mobile-model/saved_model')

# Load the image to be predicted on
image = Image.open('./workspace/training_demo/images.jpeg')

# Preprocess the image
image = np.array(image)
image = image / 255.0

# Make a prediction
predictions = model.predict(image)

# Print the predictions
print(predictions)
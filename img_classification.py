import keras
from PIL import Image, ImageOps
import numpy as np


def image_classification(img, weights_file):

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    """Create the array of the right shape to keras model"""
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    """turn the image into a numpy array"""

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    model = keras.models.load_model(weights_file, compile=False)
    """the model(weights_file) has to be in the same folder as the app"""

    prediction = model.predict(data)
    return np.argmax(prediction)
    """ run the prediction, then returns position of the highest probability"""

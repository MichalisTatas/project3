import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def extractData(filename):

    with open(filename, "rb") as f:
        f.read(4)
        num_images = int.from_bytes(f.read(4), "big", signed=True)
        x = int.from_bytes(f.read(4), "big", signed=True)
        y = int.from_bytes(f.read(4), "big", signed=True)
        buf = f.read(num_images * x * y)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, x, y).astype(np.float32)
        f.close()

        return data, x, y


def extractLabels(filename):

    with open(filename, "rb") as f:
        f.read(4)
        num_images = int.from_bytes(f.read(4), "big", signed=True)
        buf = f.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
        labels = labels.reshape(num_images).astype(np.float32)
        f.close()

        return labels

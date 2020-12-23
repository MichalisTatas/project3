import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.models import Model, load_model, Sequential


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MeanSquaredError, AUC, Accuracy, categorical_crossentropy, binary_crossentropy



from util import extractData, extractLabels

if len(sys.argv) != 2:
    print("Correct usage : python blottleneck.py + <path to train file> ")
    sys.exit(-1)

inputFile = str(sys.argv[1])

data, x, y = extractData(inputFile)

inChannel = 1
data = data.reshape(-1, x, y, inChannel)
data = data / np.max(data)


def getParameters():
    try:
        batch_size = int(input("Please enter batch size : "))
    except ValueError:
        print("batch size must be an integer")

    return batch_size

def getBottleneck():

    activationFunction="relu"
    lastActivationFunction="relu"
    # lossFunction="mean_squared_error"
    lossFunction="binary_crossentropy"
    filters=(3, 3)


    input_img = Input(shape=(x, y, inChannel))

    model = Sequential([
        Conv2D(32, filters, activation=activationFunction, padding="same", input_shape=(x, y, inChannel)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, filters, activation=activationFunction, padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, filters, activation=activationFunction, padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(10, activation=activationFunction),
        Dense(1152),
        Reshape((3, 3, 128)),
        
        Conv2DTranspose(64, filters, activation=activationFunction, strides=(2,2)),
        Conv2DTranspose(32, filters, activation=activationFunction, strides=(2,2), padding="same"),
        Conv2DTranspose(1, filters, activation=lastActivationFunction, strides=(2,2), padding="same"),
    ])


    model.compile(loss=lossFunction, optimizer=RMSprop(), metrics="accuracy",)


    train_X, valid_X, train_Y, valid_Y = train_test_split(
        data, data, test_size=0.25, shuffle=42
    )
    
    modelHistory = model.fit(train_X,
        train_Y,
        batch_size=128,
        epochs=20,
        verbose=1,
        validation_data=(valid_X, valid_Y))

    model.summary()

    return model


model = getBottleneck()

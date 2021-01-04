import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf, indices

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, MaxPooling2D, UpSampling2D, Flatten, Reshape, Embedding
from tensorflow.keras.models import Model, load_model, Sequential

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MeanSquaredError, AUC, Accuracy, categorical_crossentropy, binary_crossentropy

from util import extractData, extractLabels, plotPrediction


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

# def getBottleneck():

#     activationFunction="relu"
#     # lastActivationFunction="relu"
#     # lossFunction="mean_squared_error"
#     lossFunction="binary_crossentropy"
#     filters=(3, 3)


#     input_img = Input(shape=(x, y, inChannel))

#     model = Sequential([
#         # Embedding(input_dim = 256, output_dim = 10, input_length = 784),

#         #encoder
#         Conv2D(32, filters, activation=activationFunction, padding="same", input_shape=(x, y, inChannel)),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(64, filters, activation=activationFunction, padding="same"),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(128, filters, activation=activationFunction, padding="same"),
#         MaxPooling2D(pool_size=(2, 2)),

#         Flatten(),
#         Dense(10, activation=activationFunction),


#         #decoder
#         Dense(1152, activation=activationFunction),
#         Reshape((3, 3, 128)),

#         Conv2DTranspose(64, filters, activation=activationFunction, strides=(2,2)),
#         Conv2DTranspose(32, filters, activation=activationFunction, strides=(2,2), padding="same"),
#         Conv2DTranspose(1, filters, activation=activationFunction, strides=(2,2), padding="same"),
#     ])


#     model.compile(loss=lossFunction, optimizer=RMSprop(), metrics="accuracy",)


#     train_X, valid_X, train_Y, valid_Y = train_test_split(
#         data, data, test_size=0.25, shuffle=42
#     )
    
#     modelHistory = model.fit(train_X,
#         train_Y,
#         batch_size=128,
#         epochs=10,
#         verbose=1,
#         validation_data=(valid_X, valid_Y))

#     model.summary()

#     return model

def getAutoEncoder(filters, activationFunction):

    input_img = Input(shape=(x, y, inChannel))

    def encoder(input_img, filters):
        conv1 = Conv2D(32, filters, activation=activationFunction, padding="same")(input_img)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, filters, activation=activationFunction, padding="same")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, filters, activation=activationFunction, padding="same")(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        flat = Flatten()(pool3)

        dense = Dense(10, activation=activationFunction)(flat)

        return dense

    def decoder(input_layer, filters):
        dense = Dense(1152, activation=activationFunction)(input_layer)

        reshape = Reshape((3, 3, 128))(dense)

        convTr1 = Conv2DTranspose(64, filters, activation=activationFunction, strides=(2,2))(reshape)

        convTr2 = Conv2DTranspose(32, filters, activation=activationFunction, strides=(2,2), padding="same")(convTr1)
    
        convtr3 = Conv2DTranspose(1, filters, activation=activationFunction, strides=(2,2), padding="same")(convTr2)

        return convtr3

    def autoEncoder(input_img, filters):
        return decoder(encoder(input_img, filters), filters)

    enc = Model(input_img, encoder(input_img, filters))

    autoEnc = Model(input_img, decoder(encoder(input_img, filters), filters))

    return autoEnc, enc


def trainAutoEncoder():
    activationFunction="relu"
    # lastActivationFunction="relu"
    # lossFunction="mean_squared_error"
    lossFunction="binary_crossentropy"
    filters=(3, 3)

    autoEncoder , encoder= getAutoEncoder(filters, activationFunction)

    autoEncoder.compile(loss=lossFunction, optimizer=RMSprop(), metrics="accuracy")

    train_X, valid_X, train_Y, valid_Y = train_test_split(data, data, test_size=0.25, shuffle=42)

    modelHistory = autoEncoder.fit(train_X,
        train_Y,
        batch_size=128,
        epochs=10,
        verbose=1,
        validation_data=(valid_X, valid_Y))

    autoEncoder.summary()

    autoEncoder.save("trained")

    return autoEncoder, encoder


def loadModel():

    autoEncoder, encoder = getAutoEncoder((3,3), "relu")

    autoEncoder = load_model("trained")

    return autoEncoder, encoder


def encoderPredictions(encoder):

    min = -float(Inf)
    max = float(Inf)

    predictions = encoder.predict(data)

    # print(type(predictions))

    max = predictions.max()
    min = predictions.min()
    print(max)
    print(min)
    old_range = max - min

    new_max = 25500
    new_min = 0
    new_range = 25500


    for j, x in enumerate(predictions):
        for i, y in enumerate(x):
            predictions[j][i] = (((y - min) * new_range) / old_range) + new_min
        
        predictions[j] = predictions[j].astype(int)

    predictions=predictions.astype(int)

    # for x in predictions:
    #     print("\n", x)
    # print(type(predictions))
    # print(type(predictions[1]))
    # print(type(predictions[1][1]))

    






if __name__=="__main__":

    while True:
        try:
            answer = int(input("Press 1 to train new model\nPress 2 to load existing model\nPress 3 to exit\n"))
        except ValueError:
            print("Answer must be an integer")

        if answer == 1: # new model
            print("training new model...\n")
            autoEncoder, encoder = trainAutoEncoder()
            # plotPrediction(autoEncoder, data)
            encoderPredictions(encoder)

        elif answer == 2: # load existing model 
            print("loading existing model...\n")
            autoEncoder, encoder = loadModel()
            # plotPrediction(autoEncoder, data)
            encoderPredictions(encoder)

        elif answer == 3: # exit program
            print("exiting...\n")
            break

        else:
            print("Invalid input")
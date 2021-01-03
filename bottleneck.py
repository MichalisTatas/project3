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
    # lastActivationFunction="relu"
    # lossFunction="mean_squared_error"
    lossFunction="binary_crossentropy"
    filters=(3, 3)


    input_img = Input(shape=(x, y, inChannel))

    model = Sequential([

        #encoder
        Conv2D(32, filters, activation=activationFunction, padding="same", input_shape=(x, y, inChannel)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, filters, activation=activationFunction, padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, filters, activation=activationFunction, padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(10, activation=activationFunction),


        #decoder
        Dense(1152, activation=activationFunction),
        Reshape((3, 3, 128)),

        Conv2DTranspose(64, filters, activation=activationFunction, strides=(2,2)),
        Conv2DTranspose(32, filters, activation=activationFunction, strides=(2,2), padding="same"),
        Conv2DTranspose(1, filters, activation=activationFunction, strides=(2,2), padding="same"),
    ])


    model.compile(loss=lossFunction, optimizer=RMSprop(), metrics="accuracy",)


    train_X, valid_X, train_Y, valid_Y = train_test_split(
        data, data, test_size=0.25, shuffle=42
    )
    
    modelHistory = model.fit(train_X,
        train_Y,
        batch_size=128,
        epochs=10,
        verbose=1,
        validation_data=(valid_X, valid_Y))

    model.summary()

    return model





# def plotPrediction(model, data):
#     predictions = model.predict(data)

#     n = 10
#     plt.figure(figsize=(20, 4))
#     for i in range(1, n + 1):
#         # Display original
#         ax = plt.subplot(2, n, i)
#         plt.imshow(data[i].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)

#         # Display reconstruction
#         ax = plt.subplot(2, n, i + n)
#         plt.imshow(predictions[i].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.savefig("autoencoder.png")
#     plt.show()
#     plt.show()
#     plt.clf()
#     plt.cla()
#     plt.close()



model = getBottleneck()
predictions = model.predict(data)
# plt.imshow(predictions[1].reshape(28, 28))
# plt.gray()
# plt.show()


print(data.size)
print(predictions.size)


m = model.get_layer(index=8)

weights = m.get_weights()

# weights = np.array(weights)[indices.astype(float)]

min = float(Inf)
max = -float(Inf)

for i in weights:
    if i.max() > max:
        max = i.max()
    if i.min() < min:
        min = i.min()

print(max)
print(min)
old_range = max - min

new_max = 25500
new_min = 0
new_range = 25500

for x in weights:
    for y in x:
        print(y)
        n = (((y - min) * new_range) / old_range) + new_min
        print(n)
        

print(weights)

# flat = m.weights[0].flatten()
# print(flat)


print(type(weights[0]))
# old_range = m.weights









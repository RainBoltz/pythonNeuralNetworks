from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.datasets import mnist
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils, to_categorical
import numpy as np

def DNN(width, height, depth, classes):
    model = Sequential()

    # CONV => RELU => POOL
    model.add(Convolution2D(20, 3, 3, border_mode="same",
        input_shape=(height, width, depth)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # CONV => RELU => POOL
    model.add(Convolution2D(800, 5, 5, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # FC => RELU layers
    model.add(Flatten())
    model.add(Dense(150))
    model.add(Activation("relu"))

    # softmax
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model
    

def main():
    print("[INFO] loading DATA...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print("[INFO] compiling model...")
    model = DNN(width=28, height=28, depth=1, classes=10)
    
    # results of SGD(lr=1e-3), epoch=15: loss: 0.1914, accuracy: 94.62%
    # results of SGD(lr=1e-2), epoch=10: loss: 0.0561, accuracy: 98.29%
    # results of Adadelta(), epoch=20: loss: 0.0282, accuracy: 99.24%
    # results of Adadelta(), epoch=10: loss: 0.02xx, accuracy: 99.32%
   
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-2), metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=128, nb_epoch=10, verbose=1)

    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print("\n[INFO]loss: {:.4f}, accuracy: {:.2f}%".format(loss, accuracy * 100))
    
if __name__=="__main__":
    main()
from keras.models import Sequential
from keras.initializers import RandomUniform
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense

def DNN(width, height, classes, depth=1, filter_sizes=[20,800]):
    model = Sequential()

    # CONV => RELU => POOL
    model.add(Conv2D(filter_sizes[0], 3, 3, border_mode="same",
        input_shape=(height, width, depth)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # CONV => RELU => POOL
    model.add(Conv2D(filter_sizes[1], 5, 5, border_mode="same"))
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
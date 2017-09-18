from keras.datasets import mnist
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils, to_categorical
from keras.preprocessing import image as keras_image
from image_normalize import width_normalize
import numpy as np


def main():
    # load mnist dataset
    print("[INFO] loading DATA...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # extract mnist digits properly
    mnist_original_size = 28
    x_train = x_train.reshape(x_train.shape[0], mnist_original_size, mnist_original_size, 1)
    x_test = x_test.reshape(x_test.shape[0], mnist_original_size, mnist_original_size, 1)
    
    # generate resized datasets
    myShape = 29
    Reshape_Sizes = [10,12,14,16,18,20,0]
    train_datasets = width_normalize(x_train, reshape_sizes, output_size=myShape)
    test_datasets = width_normalize(x_test, reshape_sizes, output_size=myShape)
    
    # convert labels into categorical form
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # start compiling models
    print("[INFO] compiling model...")
    K_image = keras_image.ImageDataGenerator(rotation_range=7.5,
                                                shear_range=7.5,
                                                zoom_range=0.85)
    model = DNN(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3], classes=10)
    
    #from paper, we can know that epoch size = N, 0.001*0.993^N=0.00003, so N = 827
    model.compile(loss="categorical_crossentropy", optimizer=SGD(1e-3), metrics=["accuracy"])
    model.fit_generator(x_train, y_train, batch_size=128, nb_epoch=827, verbose=1)

    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print("\n[INFO]loss: {:.4f}, accuracy: {:.2f}%".format(loss, accuracy * 100))
    
if __name__=="__main__":
    main()
from keras.datasets import mnist
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils, to_categorical
from keras.preprocessing import image as keras_image
from image_preprocess import width_normalize
from mcdnn import MCDNN
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
    train_datasets = width_normalize(x_train, reshape_sizes=Reshape_Sizes, output_size=myShape)
    test_datasets = width_normalize(x_test, reshape_sizes=Reshape_Sizes, output_size=myShape)
    np.array(train_datasets).dump('train_datasets.pkl')
    np.array(train_datasets).dump('test_datasets.pkl')
    
    # convert labels into categorical form
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    np.array(y_train).dump('y_train.pkl')
    np.array(y_test).dump('y_test.pkl')

    # start compiling models
    print("[INFO] compiling model #%d..."%())
    
    
if __name__=="__main__":
    main()
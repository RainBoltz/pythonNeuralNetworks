from keras.datasets import mnist
from keras.utils import np_utils, to_categorical
from sys import argv
from image_preprocess import width_normalize
from mcdnn import MCDNN
import numpy as np


def main():
    print("[INFO] loading DATA...")
    if len(argv) > 1:
        # load pickles
        train_datasets = np.load(argv[1]).item()
        test_datasets = np.load(argv[2]).item()
        y_train = np.load(argv[3])
        y_test = np.load(argv[4])
    else:
        # load mnist dataset
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
        np.save('train_datasets.npy', train_datasets)
        np.save('test_datasets.npy', test_datasets)
    
        # convert labels into categorical form
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)

    # start compiling models
    MultiColumnDNN = MCDNN(train_datasets, y_train)
    MultiColumnDNN.create_columns()
    MultiColumnDNN.train_all()
    
    
if __name__=="__main__":
    main()
    
    
    
    
    
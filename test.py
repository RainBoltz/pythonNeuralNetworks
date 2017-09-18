import numpy as np
from keras.preprocessing import image as keras_image
from skimage.transform import resize as skresize
from keras.datasets import mnist
import scipy.misc as misc

(A, _), (_, _) = mnist.load_data()
mnist_original_size = 28
A = A.reshape(A.shape[0], mnist_original_size, mnist_original_size)
A = np.array(A[50])

img = misc.toimage(A)
img.show()

w = 14
h = 29
A = A.astype('float32')
A /= 255
nd = skresize(A, (h, w))
padding = 29 - w
left_padding = round(padding/2)
right_padding = padding - left_padding
A = np.pad(nd, ((0,0),(left_padding, right_padding)), mode='constant')


img2 = misc.toimage(A)
img2.show()

k = keras_image.ImageDataGenerator(featurewise_std_normalization=True)
A = A.reshape(1,29,29,1)
k.fit(A)
for x,y in k.flow(A,[0]):
    img3 = misc.toimage(x[0])
    img3.show()
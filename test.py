import numpy as np
from keras.preprocessing import image as keras_image
from skimage.transform import resize as skresize
from keras.datasets import mnist
import scipy.misc as misc
from image_preprocess import elastic_transform

(A, _), (_, _) = mnist.load_data()
mnist_original_size = 28
A = A.reshape(A.shape[0], mnist_original_size, mnist_original_size)
A = np.array(A[50])

img = misc.toimage(A)
#img.show()

#w = 14
#h = 29
#A = A.astype('float32')
#A /= 255
#nd = skresize(A, (h, w))
#padding = 29 - w
#left_padding = round(padding/2)
#right_padding = padding - left_padding
#A = np.pad(nd, ((0,0),(left_padding, right_padding)), mode='constant')


#img2 = misc.toimage(A)
#img2.show()

k = keras_image.ImageDataGenerator(featurewise_std_normalization=True,
                                    rotation_range=7.5,
                                    shear_range=0.130875,
                                    zoom_range=0.15,
                                    preprocessing_function=elastic_transform)
A = A.reshape(1,28,28,1)
k.fit(A)
flag = 0
xx = []
for x,y in k.flow(A,[0]):
    if flag == 10:
        break
    flag += 1
    img3 = misc.toimage(x[0,:,:,0])
    img3.show()
    xx.append(x)

for i in range(len(xx)-1):
    for j in range(i+1,len(xx)):
        if np.array_equal(xx[i],xx[j]):
            print("%d = %d"%(i,j))
    



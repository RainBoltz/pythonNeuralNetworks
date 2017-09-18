import numpy as np
from dnn import DNN
from keras.datasets import mnist
from keras.utils import np_utils, to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
mnist_original_size = 28
x_train = x_train.reshape(x_train.shape[0], mnist_original_size, mnist_original_size, 1)
x_test = x_test.reshape(x_test.shape[0], mnist_original_size, mnist_original_size, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model1 = DNN(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3], classes=10)
model1.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
model1.fit(x_train, y_train, nb_epoch=5, verbose=1)

loss, accuracy = model1.evaluate(x_test, y_test, verbose=1)
print('\nloss: {:.2f}%, accuracy: {:.2f}%'.format(loss*100, accuracy*100))
    
m1 = model1.predict_proba(x_test)



model2 = DNN(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3], classes=10)
model2.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
model2.fit(x_train, y_train, nb_epoch=5, verbose=1)

loss, accuracy = model2.evaluate(x_test, y_test, verbose=1)
print('\nloss: {:.2f}%, accuracy: {:.2f}%'.format(loss*100, accuracy*100))
    
m2 = model2.predict_proba(x_test)

print("m1=m2...?")
print(np.equal(m1,m2))
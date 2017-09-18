from dnn import DNN
from image_preprocess import elastic_transform

class MCDNN:
	def __init__(self, x_train, y_train):
		self.x_train = x_train
        self.y_train = y_train
        self.duplicated_models = [DNN(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3], classes=10) for _ in range(5)]
    
    def train_all(self):
        K_image = keras_image.ImageDataGenerator(featurewise_std_normalization=True,
                                                    rotation_range=7.5, #7.5~15.0 degrees
                                                    shear_range=0.130875, #7.5 degrees = 0.130875 radians
                                                    zoom_range=0.15, # y=[15,20], resize_ratio=[1-zoom_range, 1+zoom_range], zoom_range=y/100
                                                    preprocessing_function=elastic_transform)
        K_image.fit(x_train)
        model = DNN(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3], classes=10)
    
        #from paper, if use SGD(lr=1e-3), we can let epoch size as N, 0.001*0.993^N=0.00003, thus N = 827
        model.compile(loss="categorical_crossentropy", optimizer=Adadelta(), metrics=["accuracy"])
        model.fit_generator(K_image.flow(x_train, y_train, batch_size=128), steps_per_epoch=round(x_train.shape[0] / 128), nb_epoch=25, verbose=1)

        print("[INFO] evaluating...")
        (loss, accuracy) = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
        print("\n[INFO]loss: {:.4f}, accuracy: {:.2f}%".format(loss, accuracy * 100))
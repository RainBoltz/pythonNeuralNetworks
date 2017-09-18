from dnn import DNN
from keras.optimizers import SGD, Adadelta
from keras.preprocessing import image as keras_image
from image_preprocess import elastic_transform

class MCDNN:
    def __init__(self, train_datasets, y_train):
        self.train_datasets = train_datasets
        self.y_train = y_train
        self.dnns = {}
        self.Ws = list(train_datasets.keys())
    
    def create_columns(self, columns=5):
        for w in self.Ws:
            x_train = self.train_datasets[w]
            self.dnns[w] = []
            for i in range(columns):
                model = DNN(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3], classes=10) 
                model.compile(loss="categorical_crossentropy", optimizer=Adadelta(), metrics=["accuracy"])
                self.dnns[w].append(model)
    
    def train_specific(self, w_id):
        split_n = round(len(self.y_train)*0.2)
        x_train, x_test = self.train_datasets[w_id][:-split_n,:,:,:], self.train_datasets[w_id][-split_n:,:,:,:]
        y_train, y_test = self.y_train[:-split_n], self.y_train[-split_n:]
        for m in range(len(self.dnns[w_id])):
            print("[INFO] compiling model-W%d..."%(m))
            image_generator = keras_image.ImageDataGenerator(featurewise_std_normalization=True,
                                                            rotation_range=7.5, #7.5~15.0 degrees
                                                            shear_range=0.130875, #7.5 degrees = 0.130875 radians
                                                            zoom_range=0.15, # y=[15,20], resize_ratio=[1-zoom_range, 1+zoom_range], zoom_range=y/100
                                                            preprocessing_function=elastic_transform)
            image_generator.fit(x_train)
            
            #from paper, if use SGD(lr=1e-3), we can let epoch size as N, 0.001*0.993^N=0.00003, thus N = 827
            self.dnns[w_id][m].fit_generator(image_generator.flow(x_train, y_train, batch_size=128), 
                                steps_per_epoch=round(x_train.shape[0] / 128), nb_epoch=25, verbose=1, use_multiprocessing=True)

            print("[INFO] evaluating model-W%d..."%(m))
            (loss, accuracy) = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
            print("\n[INFO]loss: {:.2f}%, accuracy: {:.2f}%".format(loss * 100, accuracy * 100))
        
    def train_all(self):
        for k in self.Ws:
            self.train_specific(k)
            
            
        
        
        
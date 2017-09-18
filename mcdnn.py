from dnn import DNN
from keras.preprocessing import image as keras_image
from keras.models import load_model
from image_preprocess import elastic_transform
import os, errno

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
                model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
                self.dnns[w].append(model)
    
    def train_specific(self, w_id):
        split_n = round(len(self.y_train)*0.2)
        x_train, x_test = self.train_datasets[w_id][:-split_n,:,:,:], self.train_datasets[w_id][-split_n:,:,:,:]
        y_train, y_test = self.y_train[:-split_n], self.y_train[-split_n:]
        for m in range(len(self.dnns[w_id])):
            print("[INFO] compiling model-W%d-%d..."%(w_id, m))
            image_generator = keras_image.ImageDataGenerator(featurewise_std_normalization=True,
                                                            rotation_range=7.5, #7.5~15.0 degrees
                                                            shear_range=0.130875, #7.5 degrees = 0.130875 radians
                                                            zoom_range=0.15, # y=[15,20], resize_ratio=[1-zoom_range, 1+zoom_range], zoom_range=y/100
                                                            preprocessing_function=elastic_transform)
            image_generator.fit(x_train)
            
            #from paper, if use SGD(lr=1e-3), we can let epoch size as N, 0.001*0.993^N=0.00003, thus N = 827
            self.dnns[w_id][m].fit_generator(image_generator.flow(x_train, y_train, batch_size=128), 
                                steps_per_epoch=round(x_train.shape[0] / 128), nb_epoch=25, verbose=1)

            print("[INFO] evaluating model-W%d..."%(m))
            (loss, accuracy) = self.dnns[w_id][m].evaluate(x_test, y_test, batch_size=128, verbose=1)
            print("\n[INFO]loss: {:.2f}%, accuracy: {:.2f}%".format(loss * 100, accuracy * 100))
        
    def train_all(self):
        for w in self.Ws:
            self.train_specific(w)
        print("[INFO] DONE TRAININGS!")
            
    def output_weights(self):
        root_dir = "columns"
        try:
            os.makedirs(root_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        for w in self.Ws:
            dnn_dir = "%s/%d"%(root_dir, w)
            try:
                os.makedirs(dnn_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            for i in range(len(self.dnns[w])):
                self.dnns[w][i].model.save("%s/%d.hdf5"%(dnn_dir, i))
        print("[INFO] DONE OUTPUTING!")
        
    def load_models(self, filepath='columns'):
        W_list = os.listdir(filepath)
        for w in W_list:
            self.dnns[int(w)] = []
            M_list = os.listdir('%s/%s'%(filepath, w))
            for m in M_list:
                model = load_model('%s/%s/%s'%(filepath, w, m))
                self.dnns[int(w)].append(model)
        print("[INFO] DONE LOADING MODELS!")
        
        
    def total_evaluation(self, x_test, y_test):
        _i,_j = np.array(y_test).shape
        total_n = _i
        all_proba = np.zeros((_i, _j))
        for w in self.Ws:
            for i in range(len(self.dnns[w])):
                proba = self.dnns[w][i].predict_proba(x_test)
                all_proba = np.add(all_proba, proba)
        correct_n = 0
        for n in all_proba.shape[0]:
            pred_ans, _ = max(enumerate(all_proba[n]), key=lambda p: p[1])
            verify_ans, _ = max(enumerate(y_test[n]), key=lambda p: p[1])
            if pred_ans==verify_ans:
                correct += 1
        
        print("total: {}, corrects: {}, error-ratio: {:.3f%}".format(total_n, correct_n, (total_n-correct_n)*100/total_n))
        
        
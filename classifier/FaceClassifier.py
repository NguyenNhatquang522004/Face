import os
import pickle
import numpy as np
from sklearn import neighbors, svm

BASE_DIR = os.path.dirname(__file__) + '/'
PATH_TO_PKL = r'C:\Dataset\FaceRecognition\classifier\trained_classifier.pkl'


class FaceClassifier:
    def __init__(self, model_path=None):

        self.model = None
        # Load models
        with open(r'C:\Dataset\FaceRecognition\classifier\trained_classifier.pkl', 'rb') as f:
            self.model = pickle.load(f)
    def train(self, X, y, model='knn', save_model_path=None):
        print(save_model_path)
        if model == 'knn':
            self.model = neighbors.KNeighborsClassifier(3, weights='uniform')
        else:  # svm
            self.model = svm.SVC(kernel='linear', probability=True)
        self.model.fit(X, y)
        if save_model_path is not None:
            with open(save_model_path, 'wb') as f:
                pickle.dump(self.model, f)
    def classify(self, descriptor):
        if self.model is None:
            print('Train the model before doing classifications.')
            return

        return self.model.predict([descriptor])[0]
import joblib
import numpy as np
from sklearn.svm import LinearSVC

# Load stored data
X_train = np.load('data/ImageTrainHOG_input.npy')
y_train = np.load('data/DiseaseAugment_input.npy')
print(X_train.shape)
print(y_train.shape)

X_test = np.load('data/ImageTestHOG_input.npy')
y_test = np.load('data/DiseaseTest_input.npy')
print(X_test.shape)
print(y_test.shape)

# Classifier
svm_model = LinearSVC(C=0.01)
svm_model.fit(X_train, y_train)
print(svm_model.score(X_test, y_test))

filename = 'models/SVM_model_2.sav'
joblib.dump(svm_model, filename)

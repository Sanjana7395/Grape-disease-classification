import joblib
import numpy as np
from sklearn.svm import LinearSVC

# Load stored data
X_train = np.load('ImageTrainHOG_input.npy')
y_train = np.load('DiseaseTrain_input.npy')
print(X_train.shape)
print(y_train.shape)

X_test = np.load('ImageTestHOG_input.npy')
y_test = np.load('DiseaseTest_input.npy')
print(X_test.shape)
print(y_test.shape)

# Classifier
C = [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
for i in C:
    svm_model = LinearSVC(C=i)
    svm_model.fit(X_train, y_train)
    print(svm_model.score(X_train, y_test))

filename = 'SVM_model.sav'
joblib.dump(svm_model, filename)

from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Load stored data
X_train = np.load('ImageTrainHOG_input.npy')
y_train = np.load('Disease_input.npy')
print(X_train.shape)
print(y_train.shape)

X_test = np.load('ImageTestHOG_input.npy')
y_test = np.load('DiseaseTest_input.npy')
print(X_test.shape)
print(y_test.shape)

# Classifier
r = [20, 30, 40, 50]
for i in r:
    Random_classifier = RandomForestClassifier(n_estimators=500, max_depth=i)
    Random_classifier.fit(X_train, y_train)
    print(Random_classifier.score(X_test, y_test))

filename = 'Random_model.sav'
joblib.dump(Random_classifier, filename)

from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

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
Random_classifier = RandomForestClassifier(n_estimators=500, max_depth=35, n_jobs=-1, warm_start=True, oob_score=True,
                                           max_features='sqrt')

Random_classifier.fit(X_train, y_train)
print(Random_classifier.score(X_test, y_test))

filename = 'models/Random_model.sav'
joblib.dump(Random_classifier, filename)

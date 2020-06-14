import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocessing.utils import make_folder


def main():
    """ Load data.
    Train random forest model.
    Print accuracy on test data.

    """
    # Load stored data
    X_train = np.load('../data/processed/ImageTrainHOG_input.npy')
    y_train = np.load('../data/augment/DiseaseAugment_input.npy')
    print("=== TRAIN DATA ===")
    print(X_train.shape)
    print(y_train.shape)

    X_test = np.load('../data/processed/ImageTestHOG_input.npy')
    y_test = np.load('../data/test/DiseaseTest_input.npy')
    print("=== TEST DATA ===")
    print(X_test.shape)
    print(y_test.shape)

    # Classifier
    Random_classifier = RandomForestClassifier(n_estimators=500, max_depth=35,
                                               n_jobs=-1, warm_start=True,
                                               oob_score=True,
                                               max_features='sqrt')

    Random_classifier.fit(X_train, y_train)
    print(Random_classifier.score(X_test, y_test))

    make_folder('../results/models')
    filename = '../results/models/Random_model.sav'
    joblib.dump(Random_classifier, filename)


if __name__ == "__main__":
    main()

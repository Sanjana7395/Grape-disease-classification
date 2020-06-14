import numpy as np
from sklearn.svm import LinearSVC
import joblib
from preprocessing.utils import make_folder


def main():
    """ Load the data.
    Train SVM model using linear kernel.
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
    svm_model = LinearSVC(C=0.01)
    svm_model.fit(X_train, y_train)
    print(svm_model.score(X_test, y_test))

    make_folder('../results/models')
    filename = '../results/models/SVM_model.sav'
    joblib.dump(svm_model, filename)


if __name__ == "__main__":
    main()

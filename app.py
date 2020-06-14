import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.special import softmax
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib
from preprocessing.utils import make_folder

ROOT_DIR = 'results/models/'


def plot(image, label, index, model, x_hog):
    """ Display image, true label and predicted label.
    Save the result in the output folder

    Args:
        image (numpy array): image to predict.

        label (numpy array): true labels of corresponding images.

        index (int): index of the test image entered by the user.

        model (str): model to use. (Entered by user)

        x_hog (numpy array): feature descriptors of the image.

    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image[index])
    plt.axis('off')
    plt.title('True label: {}'.format(label[index]),
              fontdict={'fontweight': 'bold', 'fontsize': 'x-large'})
    predictions, percent = model_predict(model, image, x_hog, label)
    if model == "majority_voting":
        if predictions[index] == label[index]:
            plt.suptitle('Predicted label: {}'.format(predictions[index]),
                         color="green")
        else:
            plt.suptitle('Predicted label: {}'.format(predictions[index]),
                         color="red")

    else:
        if predictions[index] == label[index]:
            plt.suptitle('Predicted label: {} ({:.2f} %)'.format(predictions[index],
                                                                 np.max(percent[index]) * 100),
                         color="green")
        else:
            plt.suptitle('Predicted label: {} ({:.2f} %)'.format(predictions[index],
                                                                 np.max(percent[index]) * 100),
                         color="red")

    make_folder('results/visualization')
    plt.savefig('results/visualization/app.png', bbox_inches='tight')


def model_predict(model, x, hog, y):
    """ Load the given model and predict the test images

    Args:
        model (str): model as entered by the user.

        x (numpy array): images to predict.

        hog (numpy array): feature descriptors of the images.

        y (numpy array): labels of the corresponding image.

    Returns:
        predictions (numpy array): predicted labels of the corresponding image.

        percent (numpy array): Accuracy of corresponding predictions.

    """
    predictions = []
    percent = []
    labeler = LabelEncoder()
    labeler.fit(y)
    if model == "random_forest":
        rf_model = joblib.load(os.path.join(ROOT_DIR, 'Random_model.sav'))

        predictions = rf_model.predict(hog)
        percent = rf_model.predict_proba(hog)

    elif model == "svm":
        sv_model = joblib.load(os.path.join(ROOT_DIR, 'SVM_model.sav'))

        predictions = sv_model.predict(hog)
        percent = sv_model.decision_function(hog)
        percent = softmax(percent, axis=1)

    elif model == "custom_cnn":
        custom_model = load_model(os.path.join(ROOT_DIR, 'custom.h5'))

        percent = custom_model.predict(x)
        predictions = np.argmax(percent, axis=-1)
        predictions = labeler.inverse_transform(predictions)

    elif model == "vgg":
        vgg_model = load_model(os.path.join(ROOT_DIR, 'vgg16.h5'))

        percent = vgg_model.predict(x)
        predictions = np.argmax(percent, axis=-1)
        predictions = labeler.inverse_transform(predictions)

    elif model == "majority_voting":
        predictions = np.load(os.path.join(ROOT_DIR, 'Ensemble.npy'))

    elif model == "stacked_prediction":
        en_model = load_model(os.path.join(ROOT_DIR, 'custom_ensemble.h5'))

        percent = en_model.predict(np.load('data/test/X_test_ensemble.npy'))
        predictions = np.argmax(percent,
                                axis=-1)
        predictions = labeler.inverse_transform(predictions)

    return predictions, percent


def main():
    """ Predict the disease of the given image.

    Usage example:
        python app.py -m vgg -i 49

    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=int, required=True,
                    help="index of the test image")
    ap.add_argument("-m", "--model", type=str, required=True,
                    choices=("vgg", "custom_cnn",
                             "svm", "random_forest",
                             "majority_voting",
                             "stacked_prediction"),
                    help="model to be used")
    args = vars(ap.parse_args())

    X_image = np.load('data/test/ImageTest_input.npy')
    X_processed = np.load('data/processed/ImageTestHOG_input.npy')
    y = np.load('data/test/DiseaseTest_input.npy')

    plot(X_image, y, args["image"], args["model"], X_processed)


if __name__ == "__main__":
    main()

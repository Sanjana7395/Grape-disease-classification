import os
import joblib
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

ROOT_DIR = '../results/models/'


def majority_voting(rf_prediction, sv_prediction,
                    custom_prediction, vgg_prediction):
    """ Find the majority vote among the predictions of the given models

    Args:
        rf_prediction (numpy array): predictions of random forest model.

        sv_prediction (numpy array): predictions of SVM model.

        custom_prediction (numpy array): predictions of custom CNN model.

        vgg_prediction (numpy array): predictions of CNN-VGG16 model.

    Returns:
        (numpy array): predictions of ensemble-majority voting model

    """
    # loop over all predictions
    final_prediction = list()
    for rf, sv, custom, vgg in zip(rf_prediction,
                                   sv_prediction,
                                   custom_prediction,
                                   vgg_prediction):
        # Keep track of votes per class
        br = e = h = lb = pm = 0

        # Loop over all models
        image_predictions = [rf, sv, custom, vgg]
        for img_prediction in image_predictions:
            # Voting
            if img_prediction == 'black rot':
                br += 1
            elif img_prediction == 'ecsa':
                e += 1
            elif img_prediction == 'healthy':
                h += 1
            elif img_prediction == 'leaf_blight':
                lb += 1
            elif img_prediction == 'powdery mildew':
                pm += 1

        # Find max vote
        count_dict = {'br': br, 'e': e, 'h': h, 'lb': lb, 'pm': pm}
        highest = max(count_dict.values())
        max_values = [k for k, v in count_dict.items() if v == highest]
        ensemble_prediction = []
        for max_value in max_values:
            if max_value == 'br':
                ensemble_prediction.append('black rot')
            elif max_value == 'e':
                ensemble_prediction.append('ecsa')
            elif max_value == 'h':
                ensemble_prediction.append('healthy')
            elif max_value == 'lb':
                ensemble_prediction.append('leaf_blight')
            elif max_value == 'pm':
                ensemble_prediction.append('powdery mildew')

        predict = ''
        if len(ensemble_prediction) > 1:
            predict = custom
        else:
            predict = ensemble_prediction[0]

        # Store max vote
        final_prediction.append(predict)

    return np.array(final_prediction)


def main():
    """ Load data.
    Normalize and encode.
    Train ensemble-majority voting model.
    Print accuracy of the model.

    """
    X_test1 = np.load('../data/processed/ImageTestHOG_input.npy')
    X_test2 = np.load('../data/test/ImageTest_input.npy')
    y_test1 = np.load('../data/test/DiseaseTest_input.npy')
    print("=== TEST DATA ===")
    print(X_test1.shape)
    print(X_test2.shape)
    print(y_test1.shape)

    # hot encoding of labels
    labeler = LabelEncoder()
    y_test2 = labeler.fit_transform(y_test1)
    y_test2 = to_categorical(y_test2, num_classes=5)

    rf_model = joblib.load(os.path.join(ROOT_DIR, 'Random_model.sav'))
    sv_model = joblib.load(os.path.join(ROOT_DIR, 'SVM_model.sav'))
    custom_model = load_model(os.path.join(ROOT_DIR, 'custom.h5'))
    vgg_model = load_model(os.path.join(ROOT_DIR, 'vgg16.h5'))

    # Normalize image for CNN
    X_test2 = (X_test2 / 255.0).astype(np.float32)

    rf_prediction = rf_model.predict(X_test1)
    sv_prediction = sv_model.predict(X_test1)
    custom_prediction = np.argmax(custom_model.predict(X_test2), axis=-1)
    custom_prediction = labeler.inverse_transform(custom_prediction)
    vgg_prediction = np.argmax(vgg_model.predict(X_test2), axis=-1)
    vgg_prediction = labeler.inverse_transform(vgg_prediction)

    final_prediction = majority_voting(rf_prediction,
                                       sv_prediction,
                                       custom_prediction,
                                       vgg_prediction)
    # Compute accuracy
    print("ACCURACY:", accuracy_score(y_test1, final_prediction))

    # Save model
    np.save(os.path.join(ROOT_DIR, 'Ensemble.npy'), final_prediction)


if __name__ == "__main__":
    main()

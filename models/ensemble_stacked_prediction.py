import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import pickle
import joblib

NUM_CLASS = 5
ROOT_DIR = '../results/models/'


class Hist:
    """ Dummy class

    """
    def __init__(self):
        pass


def encoder(data, class_count):
    """ Transform string data to unique int.
    Convert unique int data to one-hot encoded data.

    Args:
        data (numpy array): array to be encoded.

        class_count (int): number of classes.

    Returns:
        (numpy array): one-hot encoded array.

    """
    labeler = LabelEncoder()
    y = labeler.fit_transform(data)
    y = to_categorical(y, num_classes=class_count)
    return y


def get_predictions(x_hog, x, model):
    """ Get predictions of the models

    Args:
        x_hog (numpy array): array HOG feature descriptor.

        x (numpy array): number of classes.

        model (numpy array): array of models used.

    Returns:
        (numpy array): one-hot encoded array.

    """
    rf_prediction = model[0].predict_proba(x_hog)
    sv_prediction = model[1].decision_function(x_hog)
    sv_prediction = softmax(sv_prediction, axis=1)
    custom_prediction = model[2].predict(x)
    vgg_prediction = model[3].predict(x)
    return np.concatenate([rf_prediction,
                           sv_prediction,
                           custom_prediction,
                           vgg_prediction], axis=-1)


def main():
    """ Load data.
    Normalize and encode.
    Train CNN-VGG16 model.
    Print accuracy on test data.

    """
    X_test1 = np.load('../data/processed/ImageTestHOG_input.npy')
    X_test2 = np.load('../data/test/ImageTest_input.npy')
    y_test1 = np.load('../data/test/DiseaseTest_input.npy')
    print("=== TEST DATA ===")
    print(X_test1.shape)
    print(X_test2.shape)
    print(y_test1.shape)

    X_train1 = np.load('../data/processed/ImageTrainHOG_input.npy')
    X_train2 = np.load('../data/augment/ImageAugment_input.npy')
    y_train1 = np.load('../data/augment/DiseaseAugment_input.npy')
    print("=== TRAIN DATA ===")
    print(X_train1.shape)
    print(X_train2.shape)
    print(y_train1.shape)

    # Normalize images
    X_train2 = (X_train2 / 255.0).astype(np.float32)
    X_test2 = (X_test2 / 255.0).astype(np.float32)

    # hot encoding of labels
    y_test2 = encoder(y_test1, NUM_CLASS)
    y_train2 = encoder(y_train1, NUM_CLASS)

    try:
        rf_model = joblib.load(os.path.join(ROOT_DIR, 'Random_model.sav'))
        sv_model = joblib.load(os.path.join(ROOT_DIR, 'SVM_model.sav'))
        custom_model = load_model(os.path.join(ROOT_DIR, 'custom.h5'))
        vgg_model = load_model(os.path.join(ROOT_DIR, 'vgg16.h5'))

        X_train_f = get_predictions(X_train1, X_train2,
                                    [rf_model, sv_model, custom_model, vgg_model])
        X_test_f = get_predictions(X_test1, X_test2,
                                   [rf_model, sv_model, custom_model, vgg_model])
        np.save('../data/test/X_test_ensemble.npy', X_test_f)

        # Custom CNN model
        model_custom = Sequential([
            Dense(128, input_shape=(20,)),
            Activation('relu'),
            Dropout(0.5),
            Dense(256),
            Activation('relu'),
            Dense(NUM_CLASS, activation='softmax')])

        model_custom.compile(optimizer=Adam(),
                             loss="categorical_crossentropy",
                             metrics=['accuracy'])

        # Learning rate decay
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.2,
                                                         patience=5,
                                                         min_lr=0.00001)
        history_custom = model_custom.fit(X_train_f, y_train2,
                                          batch_size=64,
                                          epochs=1, verbose=1,
                                          validation_split=.1,
                                          callbacks=[reduce_lr])
        scores = model_custom.evaluate(X_test_f, y_test2, verbose=0)
        print("========================")
        print("TEST SET: %s: %.2f%%" % (model_custom.metrics_names[1],
                                        scores[1] * 100))
        print("========================")

        print(model_custom.summary())

        # save model
        model_custom.save(os.path.join(ROOT_DIR, 'custom_ensemble.h5'))

        history = dict()
        history['acc'] = history_custom.history['acc']
        history['val_acc'] = history_custom.history['val_acc']
        history['loss'] = history_custom.history['loss']
        history['val_loss'] = history_custom.history['val_loss']

        hist = Hist()
        setattr(hist, 'history', history)
        pickle.dump(hist, open(os.path.join(ROOT_DIR, 'stacked_training_history.pkl'), 'wb'))

    except FileNotFoundError as err:
        print('[ERROR] Train random forest, SVM, CNN-custom '
              'and VGG16 models before executing ensemble model!')
        print('[ERROR MESSAGE]', err)


if __name__ == "__main__":
    main()

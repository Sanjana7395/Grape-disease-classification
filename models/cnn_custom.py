from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
import pickle
from preprocessing.utils import make_folder

NUM_CLASSES = 5


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


def main():
    """ Load data.
    Normalize and encode.
    Train custom CNN model.
    Print accuracy on test data.

    """
    # Load stored data
    X_train = np.load('../data/augment/ImageAugment_input.npy')
    y_train = np.load('../data/augment/DiseaseAugment_input.npy')
    print("=== TRAIN DATA ===")
    print(X_train.shape)
    print(y_train.shape)

    X_test = np.load('../data/test/ImageTest_input.npy')
    y_test = np.load('../data/test/DiseaseTest_input.npy')
    print("=== TEST DATA ===")
    print(X_test.shape)
    print(y_test.shape)

    # hot encoding of labels
    y_train = encoder(y_train, NUM_CLASSES)
    y_test = encoder(y_test, NUM_CLASSES)

    # Input normalization
    X_train = (X_train / 255.0).astype(np.float32)
    X_test = (X_test / 255.0).astype(np.float32)

    # Custom CNN model
    model_custom = Sequential((
        layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                      input_shape=(180, 180, 3)),
        layers.Activation('relu'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128),
        layers.Activation('relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')))

    model_custom.compile(optimizer=Adam(),
                         loss="categorical_crossentropy",
                         metrics=['accuracy'])

    # Learning rate decay
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.2,
                                                     patience=5,
                                                     min_lr=0.00001)
    history_custom = model_custom.fit(X_train, y_train, batch_size=8,
                                      epochs=1, verbose=1,
                                      validation_split=.1,
                                      callbacks=[reduce_lr])
    scores = model_custom.evaluate(X_test, y_test, verbose=0)
    print("========================")
    print("TEST SET: %s: %.2f%%" % (model_custom.metrics_names[1],
                                    scores[1] * 100))
    print("========================")

    print(model_custom.summary())

    # save model
    make_folder('../results/models/')
    model_custom.save('../results/models/custom.h5')

    history = dict()
    history['acc'] = history_custom.history['acc']
    history['val_acc'] = history_custom.history['val_acc']
    history['loss'] = history_custom.history['loss']
    history['val_loss'] = history_custom.history['val_loss']

    hist = Hist()
    setattr(hist, 'history', history)
    pickle.dump(hist, open('../results/models/custom_training_history.pkl', 'wb'))


if __name__ == "__main__":
    main()

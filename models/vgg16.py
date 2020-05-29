from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np

NUM_CLASSES = 5


class Hist():
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
    Train CNN-VGG16 model.
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

    # VGG16 CNN model
    IMG_SHAPE = (180, 180, 3)
    VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')

    VGG16_MODEL.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = Dense(NUM_CLASSES, activation='softmax')

    model_vgg16 = Sequential([
        VGG16_MODEL,
        Conv2D(512, kernel_size=(3, 3), padding='same'),
        Activation('relu'),
        Conv2D(1024, kernel_size=(3, 3), padding='same'),
        global_average_layer,
        prediction_layer
    ])

    model_vgg16.compile(optimizer=Adam(),
                        loss="categorical_crossentropy",
                        metrics=['accuracy'])

    # Learning rate decay
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.2,
                                                     patience=5,
                                                     min_lr=0.00001)
    history_custom = model_vgg16.fit(X_train, y_train, batch_size=8,
                                     epochs=20, verbose=1,
                                     validation_split=.1,
                                     callbacks=[reduce_lr])
    scores = model_vgg16.evaluate(X_test, y_test, verbose=0)
    print("========================")
    print("TEST SET: %s: %.2f%%" % (model_vgg16.metrics_names[1],
                                    scores[1] * 100))
    print("========================")

    print(model_vgg16.summary())
    print("=== BASE MODEL SUMMARY ===")
    print(VGG16_MODEL.summary())

    # save model
    model_vgg16.save('../results/models/vgg16.h5')
    history = dict()
    history['acc'] = history_custom.history['acc']
    history['val_acc'] = history_custom.history['val_acc']
    history['loss'] = history_custom.history['loss']
    history['val_loss'] = history_custom.history['val_loss']

    hist = Hist()
    setattr(hist, 'history', history)
    pickle.dump(hist, open('../results/models/vgg16_training_history.pkl', 'wb'))


if __name__ == "__main__":
    main()

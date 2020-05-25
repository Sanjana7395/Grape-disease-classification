from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle

# Load stored data
X_train = np.load('data/ImageAugment_input.npy')
y_train = np.load('data/DiseaseAugment_input.npy')
print(X_train.shape)
print(y_train.shape)

X_test = np.load('data/ImageTest_input.npy')
y_test = np.load('data/DiseaseTest_input.npy')
print(X_test.shape)
print(y_test.shape)

# hot encoding of labels
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
y_train = to_categorical(y_train, num_classes=5)
y_test = labelencoder.fit_transform(y_test)
y_test = to_categorical(y_test, num_classes=5)

# Input normalization
X_train = (X_train / 255.0).astype(np.float32)
X_test = (X_test / 255.0).astype(np.float32)

# Custom CNN model
num_classes = 5  # Number of classes to model
model_custom = Sequential([
    Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(180, 180, 3)),
    Activation('relu'),
    Conv2D(32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    Conv2D(32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    Conv2D(32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    Conv2D(32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    Conv2D(32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dropout(0.5),
    Dense(128),
    Activation('relu'),
    Dense(num_classes, activation='softmax')])

model_custom.compile(optimizer=Adam(),
                     loss="categorical_crossentropy",
                     metrics=['accuracy'])

# Learning rate decay
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                 patience=5, min_lr=0.00001)
history_custom = model_custom.fit(X_train, y_train, batch_size=8,
                                  epochs=17, verbose=1, validation_split=.1, callbacks=[reduce_lr])
scores = model_custom.evaluate(X_test, y_test, verbose=0)
print("TEST SET: %s: %.2f%%" % (model_custom.metrics_names[1], scores[1] * 100))

print(model_custom.summary())

# save model
model_custom.save('models/custom.h5')

history = dict()
history['acc'] = history_custom.history['acc']
history['val_acc'] = history_custom.history['val_acc']
history['loss'] = history_custom.history['loss']
history['val_loss'] = history_custom.history['val_loss']


class Hist():
    def __init__(self):
        pass


hist = Hist()
setattr(hist, 'history', history)
pickle.dump(hist, open('custom_training_history.pkl', 'wb'))


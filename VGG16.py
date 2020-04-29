from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Load stored data
X_train = np.load('ImageTrain_input.npy')
y_train = np.load('DiseaseTrain_input.npy')
print(X_train.shape)
print(y_train.shape)

X_test = np.load('ImageTest_input.npy')
y_test = np.load('DiseaseTest_input.npy')
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

# VGG16 CNN model
num_classes = 5  # Number of classes to model
IMG_SHAPE = (180, 180, 3)
VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

VGG16_MODEL.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = Dense(num_classes, activation='softmax')

modelvgg16 = Sequential([
    VGG16_MODEL,
    global_average_layer,
    prediction_layer
])

modelvgg16.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])

# Learning rate decay
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                 patience=5, min_lr=0.00001)
history_custom = modelvgg16.fit(X_train, y_train, batch_size=8,
                                epochs=40, verbose=1, validation_split=.1, callbacks=[reduce_lr])
scores = modelvgg16.evaluate(X_test, y_test, verbose=0)
print("TEST SET: %s: %.2f%%" % (modelvgg16.metrics_names[1], scores[1] * 100))

# Plot training & validation accuracy values
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(history_custom.history['acc'])
ax1.plot(history_custom.history['val_acc'])
ax1.set_title('Model accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
ax2.plot(history_custom.history['loss'])
ax2.plot(history_custom.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper right')

plt.savefig('results/VG1.png')

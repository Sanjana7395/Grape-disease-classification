import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from scipy.special import softmax

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation
from tensorflow.keras.optimizers import Adam

X_test1 = np.load('data/ImageTestHOG_input.npy')
X_test2 = np.load('data/ImageTest_input.npy')
y_test1 = np.load('data/DiseaseTest_input.npy')
print(X_test1.shape)
print(X_test2.shape)
print(y_test1.shape)

X_train1 = np.load('data/ImageTrainHOG_input.npy')
X_train2 = np.load('data/ImageAugment_input.npy')
y_train1 = np.load('data/DiseaseAugment_input.npy')

# Normalize images
X_train2 = (X_train2 / 255.0).astype(np.float32)
X_test2 = (X_test2 / 255.0).astype(np.float32)

# hot encoding of labels
labelencoder = LabelEncoder()
y_test2 = labelencoder.fit_transform(y_test1)
y_test2 = to_categorical(y_test2, num_classes=5)
y_train2 = labelencoder.fit_transform(y_train1)
y_train2 = to_categorical(y_train2, num_classes=5)

rf_model = joblib.load('models/Random_model.sav')
sv_model = joblib.load('models/SVM_model.sav')
custom_model = load_model('models/custom.h5')
vgg_model = load_model('models/vgg16.h5')


def get_predictions(X_HOG, X):
    rf_prediction = rf_model.predict_proba(X_HOG)
    sv_prediction = sv_model.decision_function(X_HOG)
    sv_prediction = softmax(sv_prediction, axis=1)
    custom_prediction = custom_model.predict(X)
    vgg_prediction = vgg_model.predict(X)
    return np.concatenate([rf_prediction, sv_prediction, custom_prediction, vgg_prediction], axis=-1)


X_train_f = get_predictions(X_train1, X_train2)
X_test_f = get_predictions(X_test1, X_test2)
np.save('X_test_ensemble.npy', X_test_f)

# Custom CNN model
num_classes = 5  # Number of classes to model
model_custom = Sequential([
    Dense(128, input_shape=(20,)),
    Activation('relu'),
    Dropout(0.5),
    Dense(256),
    Activation('relu'),
    Dense(num_classes, activation='softmax')])

model_custom.compile(optimizer=Adam(),
                     loss="categorical_crossentropy",
                     metrics=['accuracy'])

# Learning rate decay
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                 patience=5, min_lr=0.00001)
history_custom = model_custom.fit(X_train_f, y_train2, batch_size=64,
                                  epochs=1, verbose=1, validation_split=.1, callbacks=[reduce_lr])
scores = model_custom.evaluate(X_test_f, y_test2, verbose=0)
print("TEST SET: %s: %.2f%%" % (model_custom.metrics_names[1], scores[1] * 100))

print(model_custom.summary())

y_pred = model_custom.predict(X_test_f)
np.save('data/en2_value.npy', y_pred)
prediction = labelencoder.inverse_transform(y_pred)


# save model
model_custom.save('models/custom_ensemble.h5')

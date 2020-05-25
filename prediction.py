import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


# Display training data
def plot_images(img,labels,nrows, ncols, pred_labels=None):
    fig = plt.figure(figsize=(40, 10))
    axes = fig.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        ax.imshow(img[i])
        ax.set_xticks([]); ax.set_yticks([])
        if pred_labels is None:
            ax.set_title('True: %d' % labels[i])
        else:
            ax.set_title('True: {0}, Pred: {1}'.format(labels[i], pred_labels[i]))


X_test = np.load('data/ImageTestHOG_input.npy')
X_image = np.load('data/ImageTest_input.npy')
y_test = np.load('data/DiseaseTest_input.npy')
print(X_test.shape)
print(X_image.shape)
print(y_test.shape)

# load_model = joblib.load('models/Random_model.sav')
# load_model = joblib.load('models/SVM_model.sav')

labelencoder = LabelEncoder()
y_test1 = labelencoder.fit(y_test)
# en_model = load_model('models/custom_ensemble.h5')
# y_pred = en_model.predict(np.load('X_test_ensemble.npy'))
#########
# en_model = load_model('models/custom.h5')
# y_pred = en_model.predict(X_image)
#########
en_model = load_model('models/vgg16.h5')
y_pred = en_model.predict(np.expand_dims((X_image[0] / 255.0).astype(np.float32), axis=0))
vgg_prediction = np.argmax(y_pred, axis=-1)
classifier_prediction = labelencoder.inverse_transform(vgg_prediction)

# classifier_prediction = load_model.predict(X_test)

# plot_images(X_image, y_test, 1, 2, classifier_prediction)

plt.savefig('output/VGG16/predictions1.png')


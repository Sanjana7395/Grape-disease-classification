import numpy as np
from skimage.feature import hog


# Feature Extraction - histogram of oriented gradients
def hog_feature(image, multichannel=True):
    hog_feature_var = hog(image)
    return hog_feature_var


# Load stored data
X_train = np.load('data/ImageAugment_input.npy')
print('train data')
print(X_train.shape)

X_test = np.load('data/ImageTest_input.npy')
print('test data')
print(X_test.shape)

RF_train = np.zeros([len(X_train), 32400])
for i in range(len(X_train)):
    RF_train[i] = hog_feature(X_train[i])
print(RF_train.shape)

RF_test = np.zeros([len(X_test), 32400])
for i in range(len(X_test)):
    RF_test[i] = hog_feature(X_test[i])
print(RF_test.shape)

# Save data
np.save('data/ImageTestHOG_input.npy', RF_test)
np.save('data/ImageTrainHOG_input.npy', RF_train)

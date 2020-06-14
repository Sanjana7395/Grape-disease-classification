import numpy as np
from skimage.feature import hog
from preprocessing.utils import make_folder


def hog_feature(image, multichannel=True):
    """ Extract HOG feature descriptors from the image.

    Args:
        image (numpy array): Array of image pixels.

        multichannel (bool): True for RGB image, else False.

    Returns:
        (numpy array): Feature descriptors.

    """
    hog_feature_var = hog(image)
    return hog_feature_var


def main():
    """ Load images.
    Extract HOG feature descriptors.

    """
    # Load stored data
    X_train = np.load('../data/augment/ImageAugment_input.npy')
    print('=== TRAIN DATA ===')
    print(X_train.shape)

    X_test = np.load('../data/test/ImageTest_input.npy')
    print('=== TEST DATA ===')
    print(X_test.shape)

    print("Extracting HOG features...")
    RF_train = np.zeros([len(X_train), 32400])
    for i in range(len(X_train)):
        RF_train[i] = hog_feature(X_train[i])
    print("FEATURE DESCRIPTORS")
    print(RF_train.shape)

    RF_test = np.zeros([len(X_test), 32400])
    for i in range(len(X_test)):
        RF_test[i] = hog_feature(X_test[i])
    print(RF_test.shape)

    # Save data
    make_folder('../data/processed')
    np.save('../data/processed/ImageTestHOG_input.npy', RF_test)
    np.save('../data/processed/ImageTrainHOG_input.npy', RF_train)


if __name__ == "__main__":
    main()

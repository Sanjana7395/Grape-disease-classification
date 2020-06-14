import numpy as np
import random
from skimage import transform, exposure
from preprocessing.utils import make_folder


def random_rotation(img):
    """ Randomly rotate the image.

    Pick a random degree of rotation between.
    25% on the left and 25% on the right.

    Args:
        img (numpy array): Array of image pixels to rotate.

    Returns:
        (numpy array): Rotated image.

    """
    random_degree = random.uniform(-25, 25)
    return (transform.rotate(img, random_degree,
                             preserve_range=True)).astype(np.uint8)


def horizontal_flip(img):
    """ Flip the image horizontally.

    horizontal flip doesn't need skimage,
    it's easy as flipping the image array of pixels!

    Args:
        img (numpy array): Array of image pixels.

    Returns:
        (numpy array): Rotated image.

    """
    return img[:, ::-1]


def intensity(img):
    """ Change the intensity of the image.

    Args:
        img (numpy array): Array of image pixels.

    Returns:
        (numpy array): Rotated image.

    """
    v_min, v_max = np.percentile(img, (0.2, 99.8))
    if np.abs(v_max - v_min) < 1e-3:
        v_max += 1e-3
    return exposure.rescale_intensity(img, in_range=(v_min, v_max))


def gamma(img):
    """ Perform gamma correction of the image.

    Args:
        img (numpy array): Array of image pixels.

    Returns:
        (numpy array): Rotated image.

    """
    return exposure.adjust_gamma(img, gamma=0.4, gain=0.9)


def vertical_flip(img):
    """ Flip the image vertically.

    vertical flip doesn't need skimage,
    it's easy as flipping the image array of pixels!

    Args:
        img (numpy array): Array of image pixels.

    Returns:
        (numpy array): Rotated image.

    """
    return img[::-1, :]


def data_augment(img, y_label):
    """ Perform image augmentation using rotation,
    intensity scaling, flip and gamma correction.

    Args:
        img (numpy array): Array of image pixels.

        y_label (str): Label of the image.

    Returns:
        (numpy array): Augmented images.

        (numpy array): Array of labels corresponding to the images.

    """
    temp = [horizontal_flip(img), vertical_flip(img),
            random_rotation(img), gamma(img), intensity(img)]
    label = [y_label, y_label, y_label, y_label, y_label]
    return temp, label


def main():
    """ Load train data.
    Augment the data.

    """
    # Load data
    X_train = np.load('../data/intermediate/ImageTrain_input.npy')
    y_train = np.load('../data/intermediate/DiseaseTrain_input.npy')
    print('TO BE AUGMENTED DATA')
    print(X_train.shape)
    print(y_train.shape)

    br_count = e_count = lb_count = 0
    transformed_img = []
    y_array = []

    for i, name in enumerate(y_train):
        if name == 'healthy':
            x, y = data_augment(X_train[i], name)
            transformed_img.extend(x)
            y_array.extend(y)

        elif (name == 'black rot') and (br_count < 450):
            x, y = data_augment(X_train[i], name)
            transformed_img.extend(x)
            y_array.extend(y)
            br_count += 1

        elif (name == 'ecsa') and (e_count < 321):
            x, y = data_augment(X_train[i], name)
            transformed_img.extend(x)
            y_array.extend(y)
            e_count += 1

        elif (name == 'leaf_blight') and (lb_count < 308):
            x, y = data_augment(X_train[i], name)
            transformed_img.extend(x)
            y_array.extend(y)
            lb_count += 1

        elif name == 'powdery mildew':
            x, y = data_augment(X_train[i], name)
            transformed_img.extend(x)
            y_array.extend(y)

    transformed_img = np.array(transformed_img)
    y_array = np.array(y_array)
    print('AUGMENTED DATA')
    print(transformed_img.shape)
    print(y_array.shape)

    # Concatenate with initial image_array
    X_train = np.concatenate((X_train, transformed_img), axis=0)
    y_train = np.concatenate((y_train, y_array), axis=0)
    print('TOTAL MODEL INPUT DATA')
    print(X_train.shape)
    print(y_train.shape)

    # Save data
    make_folder('../data/augment')
    np.save('../data/augment/ImageAugment_input.npy', X_train)
    np.save('../data/augment/DiseaseAugment_input.npy', y_train)


if __name__ == "__main__":
    main()

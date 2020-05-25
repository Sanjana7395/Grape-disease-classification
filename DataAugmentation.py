import numpy as np
import random
from skimage import transform, exposure


# Boost image data set to over 10,000 images
def random_rotation(img):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return (transform.rotate(img, random_degree, preserve_range=True)).astype(np.uint8)


def horizontal_flip(img):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return img[:, ::-1]


def intensity(img):
    # change contrast
    v_min, v_max = np.percentile(img, (0.2, 99.8))
    if np.abs(v_max - v_min) < 1e-3:
        v_max += 1e-3
    return exposure.rescale_intensity(img, in_range=(v_min, v_max))


def gamma(img):
    # gamma correction
    return exposure.adjust_gamma(img, gamma=0.4, gain=0.9)


def vertical_flip(img):
    return img[::-1, :]


def data_augment1(img, y_label):
    temp = [horizontal_flip(img), vertical_flip(img), random_rotation(img), gamma(img), intensity(img)]
    label = [y_label, y_label, y_label, y_label, y_label]
    return temp, label


# Load data
X_train = np.load('data/ImageTrain_input.npy')
y_train = np.load('data/DiseaseTrain_input.npy')
print('To be augmented data')
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))

br_count = e_count = lb_count = 0
transformed_img = []
y_array = []

for i, name in enumerate(y_train):
    if name == 'healthy':
        x, y = data_augment1(X_train[i], name)
        transformed_img.extend(x)
        y_array.extend(y)

    elif (name == 'black rot') and (br_count < 450):
        x, y = data_augment1(X_train[i], name)
        transformed_img.extend(x)
        y_array.extend(y)
        br_count += 1

    elif (name == 'ecsa') and (e_count < 321):
        x, y = data_augment1(X_train[i], name)
        transformed_img.extend(x)
        y_array.extend(y)
        e_count += 1

    elif (name == 'leaf_blight') and (lb_count < 308):
        x, y = data_augment1(X_train[i], name)
        transformed_img.extend(x)
        y_array.extend(y)
        lb_count += 1

    elif name == 'powdery mildew':
        x, y = data_augment1(X_train[i], name)
        transformed_img.extend(x)
        y_array.extend(y)

transformed_img = np.array(transformed_img)
y_array = np.array(y_array)
print('Augmented data')
print(transformed_img.shape)
print(y_array.shape)
print(np.unique(y_array, return_counts=True))

# Concatenate with initial image_array
X_train = np.concatenate((X_train, transformed_img), axis=0)
y_train = np.concatenate((y_train, y_array), axis=0)
print('Total data')
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))

# Save data
np.save('data/ImageAugment_input.npy', X_train)
np.save('data/DiseaseAugment_input.npy', y_train)


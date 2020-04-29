import numpy as np
import random
from skimage import transform, exposure
from skimage import util


# Boost image data set to 10,000 images

def random_rotation(img):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return transform.rotate(img, random_degree)


def random_noise(img):
    # add random noise to the image
    return util.random_noise(img)


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


# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    'intensity': intensity,
    'gamma': gamma,
    'vertical_flip': vertical_flip
}


def data_augment(image_array, label_array, num):
    num_files_desired = num
    num_generated_files = 0
    raw_images = image_array
    transformed_image = []
    img_label = []

    while num_generated_files < num_files_desired:
        # random image from the img_arr
        index = random.randint(0, len(raw_images) - 1)
        image_path = raw_images[index]
        label_path = label_array[index]

        # random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image.append(available_transformations[key](image_path))
        img_label.append(label_path)

        num_generated_files += 1

    transformed_image = np.array(transformed_image)
    img_label = np.array(img_label)
    # Concatenate with initial image_array
    image_array = np.concatenate((image_array, transformed_image), axis=0)
    label_array = np.concatenate((label_array, img_label), axis=0)
    print(image_array.shape)
    return image_array, label_array


# Load data

X_train = np.load('ImageTrain_input.npy')
y_train = np.load('DiseaseTrain_input.npy')
print('Augmented data')
print(X_train.shape)
print(y_train.shape)

# Data augmenting

X_train, y_train = data_augment(X_train, y_train, 10000)

# Save data

np.save('Image_input.npy', X_train)
np.save('Disease_input.npy', y_train)

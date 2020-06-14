import os
import os.path
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from preprocessing.utils import make_folder

ROOT_DIR = '../data/raw/'


def get_json_data(json_path):
    """ Get data from json file and store in a table.

    The images are labelled using LabelMe tool.
    The labelled bounding box details are stored as json file.
    This function extract details from the json file in the given path.

    Args:
        json_path (str): Path of the json file.

    Returns:
        (Data frame): Contains path of the image, co-ordinates and label.

    """
    json_files = [pos_json for pos_json in os.listdir(json_path)
                  if pos_json.endswith('.json')]
    # store fields from json file in data frame
    table_data = pd.DataFrame(columns=['path', 'col1', 'col2',
                                       'row1', 'row2', 'label'])
    index = 0

    for js in json_files:
        with open(os.path.join(json_path, js)) as file:
            # load json file
            json_text = json.load(file)

            for x in json_text['shapes']:
                path = json_text['imagePath']
                points = x['points']
                # extract image section within bounding box
                if x['shape_type'] == 'rectangle':
                    col1 = int(min(points[0][1], points[1][1]))
                    col2 = int(max(points[0][1], points[1][1]))
                    row1 = int(min(points[0][0], points[1][0]))
                    row2 = int(max(points[0][0], points[1][0]))
                else:
                    col1 = int(min(points[0][1], points[3][1]))
                    col2 = int(max(points[1][1], points[2][1]))
                    row1 = int(min(points[0][0], points[1][0]))
                    row2 = int(max(points[2][0], points[3][0]))
                label = x['label']
                if label == 'black measles':
                    label = 'ecsa'

                table_data.loc[index] = [path, col1, col2, row1, row2, label]
                index += 1
    return table_data


def resize_with_aspect_ratio(img, size, interpolation):
    """ Resize image to maintain aspect ratio.

    Args:
        img (numpy array): Image to resize.

        size (int): Size to which needs to be resized.

        interpolation (str): Interpolation method to use in order to resize.

    Returns:
        (array): Resized image.

    """
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    # if h=w no padding
    if h == w:
        return cv2.resize(img, (size, size), interpolation)
    # if h!=w, make h=w by padding 0.
    if h > w:
        dif = h
    else:
        dif = w
    x_pos = int((dif - w) / 2.)
    y_pos = int((dif - h) / 2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]

    return cv2.resize(mask, (size, size), interpolation)


def get_images(img_path, js=None, valid=[".jpg", ".jpeg", ".png"], name=None):
    """ Get images from the path and store as numpy array.

    There are two sets of data set.

    1. Kaggle data set that are used as is
    2. Google images that are labelled using LabelMe tool

    Args:
        img_path (str): Path of the images folder.

        js (data frame, optional): Labelling info table.Initialized when the image is a Google images.
        Else default is None.

        valid (list): list of valid data types. Defaults are .jpeg, .jpg, .png.

        name (str): Image label. Initialized with the image label when data set used is Kaggle images.
        Else default is None. Label comes from 'js' table

    Returns:
        (numpy array): Array of desired images.

        (numpy array): Array of labels of corresponding images in the above image array.

    """
    images = []
    labels = []

    for f in os.listdir(img_path):
        # check for image files only
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid:
            continue

        # store original image
        img = plt.imread(os.path.join(img_path, f))

        # kaggle data set
        if name:
            resized_img = cv2.resize(img, (180, 180), cv2.INTER_AREA)
            images.append(resized_img)
            labels.append(name)

        # google images
        else:
            # find corresponding json files
            for index, j in enumerate(js.path):
                if j == f:
                    right_file = js.iloc[index]

                    cut = img[right_file.col1:right_file.col2,
                              right_file.row1:right_file.row2]
                    resized_img = resize_with_aspect_ratio(cut,
                                                           180,
                                                           cv2.INTER_AREA)
                    images.append(resized_img)
                    labels.append(right_file.label)
    image_arr = np.array(images)
    label_arr = np.array(labels)
    print(image_arr.shape)
    return image_arr, label_arr


def main():
    """ Load images and json files from all folders and concatenate to form a single array.
    Shuffle the array.
    Split into test and train data sets.

    """
    print('INFO: Extracting json data...')
    # Accumulate data from json files
    json_df_images = get_json_data(os.path.join(ROOT_DIR, 'images/'))
    json_df_positive = get_json_data(os.path.join(ROOT_DIR, 'positive/'))
    json_df_healthy = get_json_data(os.path.join(ROOT_DIR, 'healthy/'))
    json_df_team4 = get_json_data(os.path.join(ROOT_DIR, 'team4/'))
    json_df_team4_br = get_json_data(os.path.join(ROOT_DIR, 'team4_br/'))
    json_df_leaf_blight = get_json_data(os.path.join(ROOT_DIR, 'leaf_blight/'))

    # Accumulate data set from all folders
    print('INFO: Extracting images and corresponding labels...')
    array1, disease1 = get_images(os.path.join(ROOT_DIR, 'Grape/Black_rot/'),
                                  name='black rot')
    array2, disease2 = get_images(os.path.join(ROOT_DIR, 'Grape/Esca/'),
                                  name='ecsa')
    array3, disease3 = get_images(os.path.join(ROOT_DIR, 'Grape/Leaf_blight/'),
                                  name='leaf_blight')
    array4, disease4 = get_images(os.path.join(ROOT_DIR, 'Grape/healthy/'),
                                  name='healthy')
    array5, disease5 = get_images(os.path.join(ROOT_DIR, 'images/'),
                                  js=json_df_images)
    array6, disease6 = get_images(os.path.join(ROOT_DIR, 'positive/'),
                                  js=json_df_positive)
    array7, disease7 = get_images(os.path.join(ROOT_DIR, 'healthy/'),
                                  js=json_df_healthy)
    array8, disease8 = get_images(os.path.join(ROOT_DIR, 'team4/'),
                                  js=json_df_team4)
    array9, disease9 = get_images(os.path.join(ROOT_DIR, 'team4_br/'),
                                  js=json_df_team4_br)
    array10, disease10 = get_images(os.path.join(ROOT_DIR, 'leaf_blight/'),
                                    js=json_df_leaf_blight)

    # Concatenate data
    disease_arr = np.concatenate((disease1, disease2,
                                  disease3, disease4,
                                  disease5, disease6,
                                  disease7, disease8,
                                  disease9, disease10), axis=0)
    print('=== TOTAL DATA ===')
    print(disease_arr.shape)
    img_arr = np.concatenate((array1, array2,
                              array3, array4,
                              array5, array6,
                              array7, array8,
                              array9, array10), axis=0)
    print(img_arr.shape)

    # Shuffle data
    img_arr, disease_arr = shuffle(img_arr, disease_arr, random_state=42)
    print(np.unique(disease_arr))

    # split train set and test set
    X_train, X_test, y_train, y_test = train_test_split(img_arr, disease_arr,
                                                        test_size=0.2,
                                                        random_state=42)
    print('=== TRAIN TEST SPLIT ===')
    print(X_test.shape)
    print(X_train.shape)

    # Save data
    make_folder('../data/test')
    make_folder('../data/intermediate')
    np.save('../data/test/ImageTest_input.npy', X_test)
    np.save('../data/test/DiseaseTest_input.npy', y_test)
    np.save('../data/intermediate/ImageTrain_input.npy', X_train)
    np.save('../data/intermediate/DiseaseTrain_input.npy', y_train)


if __name__ == "__main__":
    main()

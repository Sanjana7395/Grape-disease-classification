import json
import os
import os.path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

ROOT_DIR = 'images/'


# get json files
def get_json_data(json_path):
    json_files = [pos_json for pos_json in os.listdir(json_path) if pos_json.endswith('.json')]
    # store fields from json file in data frame
    jsons_data = pd.DataFrame(columns=['path', 'col1', 'col2', 'row1', 'row2', 'label'])
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
                if label=='black measles':
                    label = 'ecsa'

                jsons_data.loc[index] = [path, col1, col2, row1, row2, label]
                index += 1
    return jsons_data


# resize the data while maintaining the aspect ratio
def resize_with_aspect_ratio(img, size, interpolation):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    # if h=w no padding
    if h == w: return cv2.resize(img, (size, size), interpolation)
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


# get the images
def get_images(img_path, js=None, valid=[".jpg", ".jpeg", ".png"], name=None):
    imgs = []
    #     original_imgs = []
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
            #             original_imgs.append(img)
            resized_img = cv2.resize(img, (180, 180), cv2.INTER_AREA)
            imgs.append(resized_img)
            labels.append(name)

        # google images
        else:
            # find corresponding json files
            for index, j in enumerate(js.path):
                if j == f:
                    #                     original_imgs.append(img)
                    right_file = js.iloc[index]

                    cut = img[right_file.col1:right_file.col2, right_file.row1:right_file.row2]
                    resized_img = resize_with_aspect_ratio(cut, 180, cv2.INTER_AREA)
                    imgs.append(resized_img)
                    labels.append(right_file.label)
    image_arr = np.array(imgs)
    label_arr = np.array(labels)
    print(image_arr.shape)
    #     return original_imgs, img_arr, label_arr
    return image_arr, label_arr


# Accumulate data from json files ###
json_df_images = get_json_data(os.path.join(ROOT_DIR, 'images/'))
json_df_positive = get_json_data(os.path.join(ROOT_DIR, 'positive/'))
json_df_healthy = get_json_data(os.path.join(ROOT_DIR, 'healthy/'))
json_df_team4 = get_json_data(os.path.join(ROOT_DIR, 'team4/'))
json_df_team4_br = get_json_data(os.path.join(ROOT_DIR, 'team4_br/'))
json_df_leaf_blight = get_json_data(os.path.join(ROOT_DIR, 'leaf_blight/'))

# Accumulate data set from all folders
print('Data in each folder')
array1, disease1 = get_images(os.path.join(ROOT_DIR, 'Grape/Black_rot/'), name='black rot')
array2, disease2 = get_images(os.path.join(ROOT_DIR, 'Grape/Esca/'), name='ecsa')
array3, disease3 = get_images(os.path.join(ROOT_DIR, 'Grape/Leaf_blight/'), name='leaf_blight')
array4, disease4 = get_images(os.path.join(ROOT_DIR, 'Grape/healthy/'), name='healthy')
array5, disease5 = get_images(os.path.join(ROOT_DIR, 'images/'), js=json_df_images)
array6, disease6 = get_images(os.path.join(ROOT_DIR, 'positive/'), js=json_df_positive)
array7, disease7 = get_images(os.path.join(ROOT_DIR, 'healthy/'), js=json_df_healthy)
array8, disease8 = get_images(os.path.join(ROOT_DIR, 'team4/'), js=json_df_team4)
array9, disease9 = get_images(os.path.join(ROOT_DIR, 'team4_br/'), js=json_df_team4_br)
array10, disease10 = get_images(os.path.join(ROOT_DIR, 'leaf_blight/'), js=json_df_leaf_blight)

# Concatenate data
disease_arr = np.concatenate((disease1, disease2, disease3, disease4, disease5, disease6, disease7, disease8, disease9,
                              disease10), axis=0)
print('Total data')
print(disease_arr.shape)
img_arr = np.concatenate((array1, array2, array3, array4, array5, array6, array7, array8, array9, array10), axis=0)
print(img_arr.shape)

# Shuffle data
img_arr, disease_arr = shuffle(img_arr, disease_arr, random_state=42)
print(np.unique(disease_arr))

# split train set and test set
X_train, X_test, y_train, y_test = train_test_split(img_arr, disease_arr, test_size=0.2, random_state=42)
print('test train split')
print(X_test.shape)
print(X_train.shape)

# Save data
np.save('data/ImageTest_input.npy', X_test)
np.save('data/DiseaseTest_input.npy', y_test)
np.save('data/ImageTrain_input.npy', X_train)
np.save('data/DiseaseTrain_input.npy', y_train)


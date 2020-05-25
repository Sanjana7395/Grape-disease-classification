import joblib
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score


X_test1 = np.load('data/ImageTestHOG_input.npy')
X_test2 = np.load('data/ImageTest_input.npy')
y_test1 = np.load('data/DiseaseTest_input.npy')
print(X_test1.shape)
print(X_test2.shape)
print(y_test1.shape)

# hot encoding of labels
labelencoder = LabelEncoder()
y_test2 = labelencoder.fit_transform(y_test1)
y_test2 = to_categorical(y_test2, num_classes=5)

rf_model = joblib.load('models/Random_model.sav')
sv_model = joblib.load('models/SVM_model.sav')
custom_model = load_model('models/custom.h5')
vgg_model = load_model('models/vgg16.h5')

# Normalize image for CNN
X_test2 = (X_test2 / 255.0).astype(np.float32)

rf_prediction = rf_model.predict(X_test1)
sv_prediction = sv_model.predict(X_test1)
custom_prediction = np.argmax(custom_model.predict(X_test2), axis=-1)
custom_prediction = labelencoder.inverse_transform(custom_prediction)
vgg_prediction = np.argmax(vgg_model.predict(X_test2), axis=-1)
vgg_prediction = labelencoder.inverse_transform(vgg_prediction)

# loop over all predictions
final_prediction = list()
for rf_pred, sv_pred, custom_pred, vgg_pred in zip(rf_prediction, sv_prediction, custom_prediction, vgg_prediction):
    # Keep track of votes per class
    br = e = h = lb = pm = 0

    # Loop over all models
    image_preds = [rf_pred, sv_pred, custom_pred, vgg_pred]
    for img_pred in image_preds:
        # Voting
        if img_pred == 'black rot':
            br += 1
        elif img_pred == 'ecsa':
            e += 1
        elif img_pred == 'healthy':
            h += 1
        elif img_pred == 'leaf_blight':
            lb += 1
        elif img_pred == 'powdery mildew':
            pm += 1

    # Find max vote
    count_dict = {'br': br, 'e': e, 'h': h, 'lb': lb, 'pm': pm}
    highest = max(count_dict.values())
    max_values = [k for k, v in count_dict.items() if v == highest]
    ensemble_prediction = []
    for max_value in max_values:
        if max_value == 'br':
            ensemble_prediction.append('black rot')
        elif max_value == 'e':
            ensemble_prediction.append('ecsa')
        elif max_value == 'h':
            ensemble_prediction.append('healthy')
        elif max_value == 'lb':
            ensemble_prediction.append('leaf_blight')
        elif max_value == 'pm':
            ensemble_prediction.append('powdery mildew')

    predict = ''
    if len(ensemble_prediction) > 1:
        predict = custom_pred
    else:
        predict = ensemble_prediction[0]

    # Store max vote
    final_prediction.append(predict)

final_prediction = np.array(final_prediction)
# Compute accuracy for final prediction
print("Accuracy:", accuracy_score(y_test1, final_prediction))

# Save model
np.save('models/Ensemble.npy', final_prediction)

np.save('data/rf_pred.npy', rf_prediction)
np.save('data/svm_pred.npy', sv_prediction)
np.save('data/cus_pred.npy', custom_prediction)
np.save('data/vgg_pred.npy', vgg_prediction)
np.save('data/en_pred.npy', final_prediction)

import joblib
import numpy as np
import yellowbrick

X_test = np.load('data/ImageTestHOG_input.npy')
y_test = np.load('data/DiseaseTest_input.npy')
print(X_test.shape)
print(y_test.shape)

rfc_model = joblib.load('models/Random_model.sav')

visualizer = yellowbrick.classifier.ROCAUC(rfc_model, classes=['healthy', 'leaf_blight', 'ecsa', 'black rot',
                                                               'powdery mildew'])
visualizer.score(X_test, y_test)
ax = visualizer.show()
ax.figure.savefig('output/random_forest/auc_roc.png')

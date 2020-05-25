import numpy as np

y = np.load('data/DiseaseTest_input.npy')
rf = np.load('data/rf_pred.npy')
sv = np.load('data/svm_pred.npy')
c = np.load('data/cus_pred.npy')
v = np.load('data/vgg_pred.npy')
en = np.load('data/en_pred.npy')
en2 = np.load('data/en2_pred.npy')

for i in range(len(y)):
    if (rf[i] != y[i]) and (sv[i] != y[i]) and (v[i] != y[i]) and (en[i] == y[i]):
        print('3 classes - 1')
        print(i)

    if (rf[i] != y[i]) and (sv[i] != y[i]) and (v[i] != y[i]) and (en[i] == y[i]) and (c[i] != y[i]):
        print('4 classes - 1')
        print(i)

    if (rf[i] != y[i]) and (sv[i] != y[i]) and (v[i] != y[i]) and (en2[i] == y[i]):
        print('3 classes - 2')
        print(i)

    if (rf[i] != y[i]) and (sv[i] != y[i]) and (v[i] != y[i]) and (en2[i] == y[i]) and (c[i] != y[i]):
        print('4 classes - 2')
        print(i)

    if(c[i] != y[i]):
        print(i)

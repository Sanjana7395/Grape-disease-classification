import numpy as np
import argparse
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=int, required=True, help="path to the input image")
ap.add_argument("-m", "--model", type=str, required=True,
                choices=("vgg", "cnn", "svm", "rf", "majority_en", "stacked_en"),
                help="model to be used")
args = vars(ap.parse_args())

X_image = np.load('data/ImageTest_input.npy')

y = np.load('data/DiseaseTest_input.npy')
rf = np.load('data/rf_pred.npy')
sv = np.load('data/svm_pred.npy')
c = np.load('data/cus_pred.npy')
v = np.load('data/vgg_pred.npy')
en = np.load('data/en_pred.npy')
en2 = np.load('data/en2_pred.npy')

rf_val = np.load('data/rf_value.npy')
sv_val = np.load('data/sv_value.npy')
c_val = np.load('data/c_value.npy')
v_val = np.load('data/vgg_value.npy')
en2_val = np.load('data/en2_value.npy')

fig = plt.figure(figsize=(8, 6))
plt.imshow(X_image[args["image"]])
plt.axis('off')
plt.title('True label: {}'.format(y[args["image"]]), fontdict={'fontweight': 'bold', 'fontsize': 'x-large'})
if args["model"] == "vgg":
    if v[args["image"]] == y[args["image"]]:
        plt.suptitle('Predicted label: {} ({:.2f} %)'.format(v[args["image"]], np.max(v_val[args["image"]])*100),
                     color="green")
    else:
        plt.suptitle('Predicted label: {} ({:.2f} %)'.format(v[args["image"]], np.max(v_val[args["image"]])*100),
                     color="red")

elif args["model"] == "cnn":
    if c[args["image"]] == y[args["image"]]:
        plt.suptitle('Predicted label: {} ({:.2f} %)'.format(c[args["image"]], np.max(c_val[args["image"]])*100),
                     color="green")
    else:
        plt.suptitle('Predicted label: {} ({:.2f} %)'.format(c[args["image"]], np.max(c_val[args["image"]])*100),
                     color="red")

elif args["model"] == "svm":
    if sv[args["image"]] == y[args["image"]]:
        plt.suptitle('Predicted label: {} ({:.2f} %)'.format(sv[args["image"]], np.max(sv_val[args["image"]])*100),
                     color="green")
    else:
        plt.suptitle('Predicted label: {} ({:.2f} %)'.format(sv[args["image"]], np.max(sv_val[args["image"]])*100),
                     color="red")

elif args["model"] == "rf":
    if rf[args["image"]] == y[args["image"]]:
        plt.suptitle('Predicted label: {} ({:.2f} %)'.format(rf[args["image"]], np.max(rf_val[args["image"]])*100),
                     color="green")
    else:
        plt.suptitle('Predicted label: {} ({:.2f} %)'.format(rf[args["image"]], np.max(rf_val[args["image"]])*100),
                     color="red")

elif args["model"] == "majority_en":
    if en[args["image"]] == y[args["image"]]:
        plt.suptitle('Predicted label: {}'.format(en[args["image"]]), color="green")
    else:
        plt.suptitle('Predicted label: {}'.format(en[args["image"]]), color="red")

elif args["model"] == "stacked_en":
    if en2[args["image"]] == y[args["image"]]:
        plt.suptitle('Predicted label: {} ({:.2f} %)'.format(en2[args["image"]], np.max(en2_val[args["image"]])*100),
                     color="green")
    else:
        plt.suptitle('Predicted label: {} ({:.2f} %)'.format(en2[args["image"]], np.max(en2_val[args["image"]])*100),
                     color="red")

plt.savefig('output/demo.png', bbox_inches='tight')

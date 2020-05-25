import pickle
import matplotlib.pyplot as plt


class Hist():
    def __init__(self):
        pass


# history_custom = pickle.load(open('vgg16_training_history.pkl', 'rb'))
history_custom = pickle.load(open('custom_training_history.pkl', 'rb'))

# Plot training & validation accuracy values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 8])
ax1.plot(history_custom.history['acc'])
ax1.plot(history_custom.history['val_acc'])
ax1.set_title('Model accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss values
ax2.plot(history_custom.history['loss'])
ax2.plot(history_custom.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper right')

# plt.savefig('output/VGG16/acc_loss.png')
plt.savefig('output/Custom/acc_loss1.png')

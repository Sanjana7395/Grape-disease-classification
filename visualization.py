import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yellowbrick
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from tensorflow.keras.models import load_model
import joblib
import pickle
from preprocessing.utils import make_folder

ROOT_DIR = 'results/models/'


class Hist:
    """ Dummy class

    """

    def __init__(self):
        pass


def visualize(visual_type, model, x, y):
    """ Execute function depending on the user input 'type'

    Args:
        visual_type (str): type of visualization technique to plot.

        model (str): model for which visualization needs to be plot.

        x (numpy array): test images.

        y (numpy array): test labels.

    """
    if visual_type == "confusion_matrix":
        con_matrix(model, x, y)

    elif visual_type == "acc_loss":
        plot(model)

    elif visual_type == "tree":
        tree()

    elif visual_type == "ROC":
        roc(x, y)


def roc(x, y):
    """ Plot ROC-AUC plot for random forest model.
    Save the image in output folder.

    Args:
        x (numpy array): test images.

        y (numpy array): test labels.

    """
    model = joblib.load(os.path.join(ROOT_DIR, 'Random_model.sav'))

    visualizer = yellowbrick.classifier.ROCAUC(model,
                                               classes=['healthy',
                                                        'leaf_blight',
                                                        'ecsa',
                                                        'black rot',
                                                        'powdery mildew'])
    visualizer.score(x, y)
    ax = visualizer.show()
    make_folder('results/visualization')
    ax.figure.savefig('results/visualization/auc_roc.png')


def tree():
    """ Plot the tree for random forest model.
    Save the dot file in output folder.
    Convert dot file to png by using the command:
    'dot -Tpng tree.dot -o tree.png'

    """
    model = joblib.load(os.path.join(ROOT_DIR, 'Random_model.sav'))
    tree_num = model.estimators_
    make_folder('results/visualization')
    for tree_in_forest in tree_num:
        export_graphviz(tree_in_forest, out_file='results/visualization/tree.dot',
                        filled=True, rounded=True,
                        precision=2)


def plot(model):
    """ Plot the accuracy and loss curve for the neural networks.
    Save file in the output folder.

    Args:
        model (str): model for which visualization needs to be plot

    """
    history_custom = Hist()
    if model == "cnn_custom":
        history_custom = pickle.load(open(os.path.join(ROOT_DIR, 'custom_training_history.pkl'),
                                          'rb'))

    elif model == "vgg":
        history_custom = pickle.load(open(os.path.join(ROOT_DIR, 'vgg16_training_history.pkl'),
                                          'rb'))

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

    make_folder('results/visualization')
    plt.savefig('results/visualization/acc_loss_{}.png'.format(model))


def con_matrix(model, x, y):
    """ Plot confusion matrix for the given model.
    Save the png in the output folder.

    Args:
        model (str): model for which visualization needs to be plot.

        x (numpy array): test images.

        y (numpy array): test labels.

    """
    corr = []
    if model == "random_forest":
        loaded_model = joblib.load(os.path.join(ROOT_DIR, 'Random_model.sav'))
        classifier_prediction = loaded_model.predict(x)
        corr = confusion_matrix(y, classifier_prediction)

    elif model == "svm":
        loaded_model = joblib.load(os.path.join(ROOT_DIR, 'SVM_model.sav'))
        classifier_prediction = loaded_model.predict(x)
        corr = confusion_matrix(y, classifier_prediction)

    elif model == "majority_voting":
        classifier_prediction = np.load(os.path.join(ROOT_DIR, 'Ensemble.npy'))
        corr = confusion_matrix(y, classifier_prediction)

    elif model == "stacked_prediction":
        labeler = LabelEncoder()
        labeler.fit(y)
        loaded_model = load_model(os.path.join(ROOT_DIR, 'custom_ensemble.h5'))
        y_prediction = loaded_model.predict(np.load('data/test/X_test_ensemble.npy'))
        prediction = np.argmax(y_prediction, axis=-1)
        prediction = labeler.inverse_transform(prediction)
        corr = confusion_matrix(y, prediction)

    make_confusion_matrix(corr,
                          categories=['blackrot', 'ecsa',
                                      'healthy', 'leafblight',
                                      'pmildew'],
                          count=True,
                          percent=True,
                          color_bar=False,
                          xy_ticks=True,
                          xy_plot_labels=True,
                          sum_stats=True,
                          fig_size=(8, 6),
                          c_map='OrRd',
                          title='Confusion matrix')
    # error correction - cropped heat map
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values

    make_folder('results/visualization')
    plt.savefig('results/visualization/confusion_matrix_{}.png'.format(model),
                bbox_inches='tight')


def make_confusion_matrix(cf, categories,
                          group_names=None,
                          count=True,
                          percent=True,
                          color_bar=True,
                          xy_ticks=True,
                          xy_plot_labels=True,
                          sum_stats=True,
                          fig_size=None,
                          c_map='Blues',
                          title=None):
    """ Code to generate text within each box and beautify confusion matrix.

    Args:
        cf (numpy array): Confusion matrix.

        categories (numpy array): array of classes.

        group_names (numpy array): classes in the project.

        count (bool): whether to display the count of each class.

        percent (bool): whether to display percentage for each class.

        color_bar (bool): whether to display color bar for the heat map.

        xy_ticks (bool): whether to display xy labels.

        xy_plot_labels (bool): whether to display xy title.

        sum_stats (bool):whether to display overall accuracy.

        fig_size (tuple): size of the plot.

        c_map (str): color scheme to use.

        title (str): Title of the plot.

    """
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        row_size = np.size(cf, 0)
        col_size = np.size(cf, 1)
        group_percentages = []
        for i in range(row_size):
            for j in range(col_size):
                group_percentages.append(cf[i][j] / cf[i].sum())
        group_percentages = ["{0:.2%}".format(value)
                             for value in group_percentages]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip()
                  for v1, v2, v3 in zip(group_labels,
                                        group_counts,
                                        group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))
        stats_text = "\n\nAccuracy={0:0.2%}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if fig_size is None:
        # Get default figure size if not set
        fig_size = plt.rcParams.get('figure.figsize')

    if not xy_ticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEAT MAP VISUALIZATION
    plt.figure(figsize=fig_size)
    sns.heatmap(cf, annot=box_labels, fmt="",
                cmap=c_map, cbar=color_bar,
                xticklabels=categories,
                yticklabels=categories)

    if xy_plot_labels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def main():
    """ Accept user input.
    Depending on the input plot the required graph.

    Usage example:
        python visualization.py -t confusion matrix -m svm

    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str, required=True,
                    choices=("confusion_matrix", "acc_loss",
                             "tree", "ROC"),
                    help="type of visualization")
    ap.add_argument("-m", "--model", type=str, required=False,
                    choices=("random_forest", "svm",
                             "majority_voting", "stacked_prediction",
                             "cnn_custom", "vgg"),
                    help="type of visualization")
    args = vars(ap.parse_args())

    X_test = np.load('data/processed/ImageTestHOG_input.npy')
    y_test = np.load('data/test/DiseaseTest_input.npy')
    print(X_test.shape)
    print(y_test.shape)

    visualize(args["type"], args["model"], X_test, y_test)


if __name__ == "__main__":
    main()

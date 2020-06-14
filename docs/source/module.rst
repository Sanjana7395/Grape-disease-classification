Documentation for the code
===========================

1. Pre processing
      This folder contains
         A. Code to load the images and json(contains labelling information) files. This is present in
            preprocessing/001_load_data.py. To execute this code, within the 'preprocessing' folder enter the below
            command. ::

                python 001_load_data.py

         B. Augment data. The code is present in preproprocessing/002_data_augmentation.py. To execute, run
            the below command. ::

                python 002_data_augmentation.py

         C. Extract histograms of feature descriptors. Feature descriptors are used to train only
            random forest and SVM. The code is present in preprocessing/003_hog.py. ::

                python 003_hog.py

2. Models
      This folder contains various models used in this project namely:
         A. Random forest
         B. Support vector machine
         C. CNN - VGG16
         D. CNN - Custom
         E. Ensemble model - Majority voting
         F. Ensemble model - Stacked prediction

      The ensemble models are the aggregation of random forest, SVM, CNN-custom and CNN-VGG16.
      The models can be trained by executing the below command within the models folder. ::

          python <model_name>.py

3. visualization.py
      This file contains all the visualization techniques used in this project.
         A. Confusion matrix, using sns heat map with modifications to display details within each box.
         B. Loss and Accuracy curves for Neural networks.
         C. Tree representation for Random forest
         D. ROC-AUC curves using Yellowbrick.

   Usage is as follows ::

         python visualization.py -m <model_name> -t <one_visualization_technique>

   For help on available models and visualization techniques ::

         python visualization.py --help

4. app.py
      This file predicts the disease of the input image. Usage is as follows ::

         python app.py -m <model_name> -i <test_image_index>

      for help on usage ::

         python app.py --help

Classification main (app.py)
----------------------------
.. automodule:: app
   :members:

Classification visualization (visualization.py)
------------------------------------------------
.. automodule:: visualization
   :members:

Pre processing - load data (001_load_data.py)
----------------------------------------------
.. automodule:: preprocessing.001_load_data
   :members:

Pre processing - data augmentation (002_data_augmentation.py)
--------------------------------------------------------------

Data augmentation techniques used are -

1. Horizontal flip
2. Vertical flip
3. Gamma correction
4. Intensity scaling
5. Random rotation

.. automodule:: preprocessing.002_data_augmentation
   :members:

Pre processing - HOG (003_hog.py)
----------------------------------

Feature descriptors are generated using Histograms of Oriented Gradients

.. automodule:: preprocessing.003_hog
   :members:

Models - CNN-Custom (cnn_custom)
---------------------------------
.. automodule:: models.cnn_custom
   :members:

Models - Majority Voting (ensemble_majority_voting.py)
-------------------------------------------------------

In majority voting technique, output prediction is the one that
receives more than half of the votes or the maximum
number of votes. If none of the predictions get more
than half of the votes or if it is a tie, we may say that
the ensemble method could not make a stable
prediction for this instance. In such a situation the
prediction of the model with the highest accuracy is
taken as the final output.

.. automodule:: models.ensemble_majority_voting
   :members:

Models - Stacked Prediction (ensemble_stacked_prediction.py)
-------------------------------------------------------------

The network is trained with the array of probabilities from all 4 models.

.. automodule:: models.ensemble_stacked_prediction
   :members:

Models - Random forest (random_forest.py)
------------------------------------------
.. automodule:: models.random_forest
   :members:

Models - SVM (svm.py)
----------------------
.. automodule:: models.svm
   :members:

Models - VGG16 (vgg16.py)
-------------------------
.. automodule:: models.vgg16
   :members:

Indices and Tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Results
========

Below are the results obtained on the test set for various models trained in the project.

.. note:: The results obtained are system specific. Due to different combinations of the neural
    network cudnn library versions and NVIDIA driver library versions, the results can be
    slightly different. To the best of my knowledge, upon reproducing the environment, the
    ballpark number will be close to the results obtained.

+----------------------------------+---------------+
| Models                           | Accuracy (%)  |
+==================================+===============+
| Random forest                    | 75.35         |
+----------------------------------+---------------+
| SVM                              | 82.89         |
+----------------------------------+---------------+
| CNN - VGG16                      | 93.62         |
+----------------------------------+---------------+
| Ensemble - Majority voting       | 98.05         |
+----------------------------------+---------------+
| Ensemble - Stacked prediction    | 98.23         |
+----------------------------------+---------------+
| CNN - Custom                     | 98.76         |
+----------------------------------+---------------+

Indices and Tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

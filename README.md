# Grape Disease Detection

This project classifies diseases in grape plant using various Machine Learning classification algorithms.
Grape plants are susceptible to various diseases The diseases that are classified in this project are:

1. Black rot
2. Black Measles (esca)
3. Powdery mildew
4. Leaf blight
5. Healthy

The Machine learning classification models in this project includes:

1. Random forest classification
2. Support vector machine classification
3. CNN - VGG16
4. CNN - Custom
5. Ensemble model - Majority voting
6. Ensemble model - Stacked prediction

Configuration of Project Environment
=====================================

1. Clone the project.
2. Install packages required.
3. Download the data set
4. Run the project.

Setup procedure
----------------
1. Clone project from [GitHub](https://github.com/Sanjana7395/Grape-disease-classification.git).  
      Change to the directory Grape-Disease-Classification.
2. Install packages  
   In order to reproduce the code install the packages 
   
   1. Manually install packages mentioned in requirements.txt file or use the command.

           pip install -r requirements.txt

   2. Install packages using setup.py file.

            python setup.py install

   The **--user** option directs setup.py to install the package
   in the user site-packages directory for the running Python.
   Alternatively, you can use the **--home** or **--prefix** option to install
   your package in a different location (where you have the necessary permissions)

3. Download the required data set.  
      The data set that is used in this project is available
      [here](https://drive.google.com/drive/folders/1SFBc-dNzr325jHw434j8LYyCii6djzkC?usp=sharing).
      The data set includes images from [kaggle](https://www.kaggle.com/xabdallahali/plantvillage-dataset)
      grape disease data set and the images collected online and labelled using the **LabelMe** tool.  
      Download the zip file and extract the files in **data/raw** folder.  
      
      [OR]
      
      Run the below command
      
        ./wgetgdrive.sh <drive_id> <zip_name>.zip
        
      drive_id is **1gsUyWEkxz9H1-yn2ONx4scHg88kWU-38**  
      Provide any zip_name.

4. Run the project.  
      See **Documentation for the code** section for further details.
      
Documentation for the code
===========================

1. __Pre processing__  
   This folder contains  
      
   1. Code to load the images and json(contains labelling information) files. This is present in
   preprocessing/001_load_data.py. To execute this code, within the 'preprocessing' folder enter the below
   command
           
           python 001_load_data.py
              
   2. Augment data. The code is present in preproprocessing/002_data_augmentation.py. To execute, run
    the below command
            
            python 002_data_augmentation.py
               
        The data augmentation techniques used are
        - Horizontal flip
        - Vertical flip
        - Random rotation
        - Intensity scaling
        - Gamma correction   
        
   3. Extract histograms of feature descriptors. Feature descriptors are used to train only
   random forest and SVM. The code is present in preprocessing/003_hog.py
   
            python 003_hog.py

2. __Models__  
   This folder contains various models used in this project namely:
   
   1. Random forest
   2. Support vector machine
   3. CNN - VGG16
   4. CNN - Custom
   5. Ensemble model - Majority voting  
In majority voting technique, output prediction is the one that
receives more than half of the votes or the maximum
number of votes. If none of the predictions get more
than half of the votes or if it is a tie, we may say that
the ensemble method could not make a stable
prediction for this instance. In such a situation the
prediction of the model with the highest accuracy is
taken as the final output.
   6. Ensemble model - Stacked prediction  
The network is trained with the array of probabilities from all 4 models.

   The ensemble models are the aggregation of random forest, SVM, CNN-custom and CNN-VGG16.
   
   The models can be trained by executing the below command within the models folder
   
          python <model_name>.py

3. __visualization.py__  
      This file contains all the visualization techniques used in this project.
   1. Confusion matrix, using sns heat map with modifications to display details within each box.
   2. Loss and Accuracy curves for Neural networks.
   3. Tree representation for Random forest
   4. ROC-AUC curves using Yellowbrick.
   
   Usage is as follows  
                    
         python visualization.py -m <model_name> -t <one_visualization_technique>
          
   For help on available models and visualization techniques
   
         python visualization.py --help

4. __app.py__  
      This file predicts the disease of the input image. Usage is as follows
      
         python app.py -m <model_name> -i <test_image_index>
         
      for help on usage
      
         python app.py --help
      
Results
========

Below are the results obtained on the test set for various models trained in the project.

> NOTE    
   The results obtained are system specific. Due to different combinations of the neural 
   network cudnn library versions and NVIDIA driver library versions, the results can be 
   slightly different. To the best of my knowledge, upon reproducing the environment, the
   ballpark number will be close to the results obtained.

| Models                           | Accuracy (%)  |
|----------------------------------|:-------------:|
| Random forest                    | 75.35         |
| SVM                              | 82.89         |
| CNN - VGG16                      | 93.62         |
| Ensemble - Majority voting       | 98.05         |
| Ensemble - Stacked prediction    | 98.23         |
| CNN - Custom                     | 98.76         |

      
      
      

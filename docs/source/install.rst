Configuration of Project Environment
=====================================

1. Clone the project.
2. Install packages required.
3. Download the data set
4. Run the project.

Setup procedure
----------------
1. Clone project from `GitHub <https://github.com/Sanjana7395/Grape-disease-classification.git>`_
      Change to the directory Grape-Disease-Classification.
2. Install packages
      In order to reproduce the code install the packages
         A. Manually install packages mentioned in requirements.txt file or use the command. ::

               pip install -r requirements.txt

         B. Install packages using setup.py file. ::

               python setup.py install

            The **---user** option directs setup.py to install the package
            in the user site-packages directory for the running Python.
            Alternatively, you can use the **---home** or **---prefix** option to install
            your package in a different location (where you have the necessary permissions)

         .. note:: The requirements.txt file replicates the virtual environment that I use. It has many packages
                  that are not relevant to this project. Feel free to edit the packages list.

3. Download the required data set.
      The data set that is used in this project is available
      `here. <https://drive.google.com/drive/folders/1SFBc-dNzr325jHw434j8LYyCii6djzkC?usp=sharing>`_
      The data set includes images from `kaggle <https://www.kaggle.com/xabdallahali/plantvillage-dataset>`_
      grape disease data set and the images collected online and labelled using the LabelMe tool.
      Download the zip file and extract the files in **data/raw** folder.

      [OR]

      Run the below command ::

        ./wgetgdrive.sh <drive_id> <zip_name>.zip

      drive_id is **1gsUyWEkxz9H1-yn2ONx4scHg88kWU-38**.
      Provide any zip_name.

4. Run the project.
      See **Documentation for the code** section for further details.

Indices and Tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

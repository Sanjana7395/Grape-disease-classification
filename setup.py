import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.
def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


def get_packages():
    requirementPath = 'requirements.txt'
    packages = []
    if os.path.isfile(requirementPath):
        with open(requirementPath) as f:
            packages = f.read().splitlines()
    return packages


setup(
    name="grape_disease_classification",
    version="1.0.0",
    author="Sanjana Srinivas",
    author_email="sanjanasrinivas73@gmail.com",
    description=("A demonstration of how to classify diseases in"
                 " plants(Grape) using various Machine Learning models."),
    keywords="disease classification",
    url="https://github.com/Sanjana7395/Grape-disease-classification.git",
    install_requires=get_packages(),
    long_description=read('README.md'),
)

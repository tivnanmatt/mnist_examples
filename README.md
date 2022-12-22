# MNIST Deep Learning Examples

## Motivation

The goal of this project is to provide examples of deep learning models applied to the MNIST dataset, which consists of images of handwritten digits. These examples can be used as a reference for learning about deep learning and for building custom models for image classification and image denoising tasks.

## Conda Installation

To install the required dependencies and set up the software environment for this project, run the following conda command:

`conda env create -f environment.yml`

## Usage

To run the code, navigate to the root directory of the project and execute the desired script using Python:

`python classifier.py`

or 

`python denoising.py`

This will execute the code and generate the output.

## Files 

- classification.py: This script trains and tests a deep learning model for image classification on the MNIST dataset.
- ddpm.py: This script trains and tests a deep learning model for image classification using a Dropout Deep Probabilistic Model (DDPM).
- denoising.py: This script trains and tests a denoising model on the MNIST dataset. The denoiser is designed to remove noise from images of handwritten digits.
- environment.yml: This file defines the conda environment for the project.
- README.md: This file provides an overview of the project.
- tmp.py: This file is a temporary script used for testing purposes.
- utils.py: This script contains helper functions and classes used by the other scripts.
- .gitignore: This file lists files and directories that should be ignored by Git.
- data/: This directory contains the MNIST dataset and related files.
- weights/: This directory contains the trained model weights.



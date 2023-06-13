# Master thesis: Detection and removal of texts burned in medical images using deep learning

- **Author**    : Szelagowski Nicolas
- **Promotor**  : Jodogne Sébastien
- **Academic year** : 2022-2023

# Project Structure

## MTC

This directory contains the source code of the MTC interface and the instructutions to compile the Vue.js project.


## OrthancPlugin

This directory contains the necessary files to integrate MedTextCleaner (MTC) into Orthanc, a python plugin designed to help users remove texts burned in medical images.


## SSD-TB

This directory contains the files used to train the models (Textboxes and SSD).

## evaluation

This directory contains the files used in order to evaluate our models, including the detection evaluation protocol scripts (icdar, deteval, coco) and a google colab notebook inspired from this [tutorial](https://docs.voxel51.com/tutorials/evaluate_detections.html#Evaluating-Object-Detections-with-FiftyOne).


## dataset_generator.ipynb

This notebook takes care of generating synthetic data on JPEG images and produces matching annotation files.

## dicomTojpeg.py

This python script allows to convert DICOM instances to JPEG images using an Orthanc Server.

## interactiveClasifier.ipynb

This notebook provides a user interface designed to simplify the manual classification of images into two categories. 





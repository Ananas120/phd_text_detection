# Purpose

MedTextCleaner is a python plugin designed to help users remove texts burned in medical images.

# Description

This demo contains:

- an Orthanc container with the Python plugin enabled
- a Server.py script that extends the Orthanc Rest API with 2 
  routes (predict, redact) and also adds 1 button in the
  Orthanc Explorer to access MTC.

# Starting the setup

To start the setup, type: `docker-compose up --build`

# demo

- open your Orthanc Explorer on [http://localhost:8000](http://localhost:8000) (username: demo, password: demo)
- upload a study
- browse to an instance, you'll now see a button "MedTextCleaer" that will redirect to the MTC user interface
- after a short period of time, rectangles will be drawn on the image where text areas have been located
- adjust the predictions if necessary
- generate a new dicom instance where text areas have been blacked out by clikcing on the "download dicom" button.

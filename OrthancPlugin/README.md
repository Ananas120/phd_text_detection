# Purpose

MedTextCleaner is a python plugin for Orthanc, designed to help users remove texts burned in medical images.

# Description

This project contains:

- an Orthanc container with the Python plugin enabled
- a Server.py script that extends the Orthanc Rest API with 2 
  routes (predict, redact) and also adds 1 button in the
  Orthanc Explorer to access MTC.
  
# Licensing

The MTC plugin for Orthanc is licensed under the GPL license.

We also kindly ask scientific works and clinical studies that make use of Orthanc to cite Orthanc in their associated publications.
Similarly, we ask open-source and closed-source products that make use of Orthanc to warn us about this use. You can cite our work using the following BibTeX entry:


@Article{Jodogne2018,
  author="Jodogne, S{\'e}bastien",
  title="The {O}rthanc Ecosystem for Medical Imaging",
  journal="Journal of Digital Imaging",
  year="2018",
  month="Jun",
  day="01",
  volume="31",
  number="3",
  pages="341--352",
  issn="1618-727X",
  doi="10.1007/s10278-018-0082-y",
  url="https://doi.org/10.1007/s10278-018-0082-y"
}

# Starting the setup

To start the setup, type: `docker-compose up --build`

# demo

- open your Orthanc Explorer on [http://localhost:8000](http://localhost:8000) (username: demo, password: demo)
- upload a study
- browse to an instance, you'll now see a button "MedTextCleaer" that will redirect to the MTC user interface
- after a short period of time, rectangles will be drawn on the image where text areas have been located
- adjust the predictions if necessary
- generate a new dicom instance where text areas have been blacked out by clikcing on the "download dicom" button.

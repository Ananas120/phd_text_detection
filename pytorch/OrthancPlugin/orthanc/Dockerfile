FROM osimis/orthanc

RUN pip3 install pydicom
RUN pip3 install Pillow
RUN pip3 install numpy
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install matplotlib

RUN mkdir /python
RUN mkdir /python/src
RUN mkdir /python/MedTextCleaner

COPY extend-explorer.js /python/extend-explorer.js
COPY Server.py /python/Server.py
COPY src /python/src
COPY MedTextCleaner /python/MedTextCleaner


"""
MedTextCleaner - An Orthanc plugin for text removal in medical images

This program is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

 
import io
import orthanc
import pydicom
from pydicom.uid import generate_uid
import pprint
import os
import base64
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import time
from src.model import Textboxes, ResNet, SSD
from src.dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from src.transform import SSDTransformer
from src.utils import generate_dboxes, Encoder


global transformer
global encoder
global model


def serve_vue_app_files(output, uri, **request):
    if request['method'] == 'GET':
        path = request['groups'][0]
        try:
            with open(os.path.join('/python/MedTextCleaner/', path), 'rb') as f:
                content = f.read()
                content_type = 'text/html'
                if path.endswith('.css'):
                    content_type = 'text/css'
                elif path.endswith('.js'):
                    content_type = 'text/javascript'
                output.AnswerBuffer(content, content_type)
        except FileNotFoundError:
            output.AnswerBuffer("File {} Not Found".format(path), 'text/plain')
    else:
        output.SendMethodNotAllowed('GET')


        
def get_predictions(image):
    conf_trhesh = 0.2
    nms_thresh = 0.5
    max_pred = 200
    
    predictions = []
    width, height = image.size
    
    start_time = time.time()
    img, _, _, _ = transformer(image, None, torch.zeros(1,4), torch.zeros(1))
    if torch.cuda.is_available():
        img = img.cuda()
    end_time = time.time()
    diff1 = (end_time - start_time)*1000
    print('Pre-processing time: {} ms'.format(diff1))

    with torch.no_grad():
        # Get predictions
        start_time = time.time()
        ploc, plabel = model(img.unsqueeze(0))
        end_time = time.time()
        diff2 = (end_time - start_time)*1000
        print('Inference time: {} ms'.format(diff2))
        
        start_time = time.time()
        ploc, plabel = ploc.float(), plabel.float()

        result = encoder.decode_batch(ploc, plabel,nms_thresh, max_pred)[0]

        loc, label, prob = [r.cpu().numpy() for r in result]

        for loc_, _, prob_ in zip(loc, label, prob):
            if prob_ > conf_trhesh:
                xmin,ymin, w, h = loc_[0]*width, loc_[1]*height, (loc_[2] - loc_[0])*width, (loc_[3] - loc_[1])*height
                bbox = [int(xmin), int(ymin), int(w), int(h)]
                predictions.append(bbox)
        end_time = time.time()
        diff3 = (end_time - start_time)*1000
        print('Post-processing time: {} ms'.format(diff3))
    print('Total time: {} ms'.format(diff1 + diff2 + diff3))
    return predictions

        
def predict(output, uri, **request):
    if request["method"] == "POST":
        # Retrieve the image data directly from the input buffer
        image_data = request["body"]
        
        # Convert the image data to a Pillow image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        boxes = get_predictions(image)
        response_data = {
            'boxes': boxes
    	}
        result_json = json.dumps(response_data)

        # Send a response back to the client with the content type set to 'application/json'
        output.AnswerBuffer(result_json, "application/json")

    else:
        output.SendMethodNotAllowed("POST")
        
def redact(output, uri, **request):

    if request["method"] == "POST":
        instanceId = request['groups'][0]

        data = json.loads(request["body"])
    
        rectangles = data["rectangles"]

        f = orthanc.GetDicomForInstance(instanceId)
        new_dicom_instance = pydicom.dcmread(io.BytesIO(f))
        pixel_data = new_dicom_instance.pixel_array
        
        img = Image.fromarray(pixel_data)

        for left, top, width, height in rectangles:
            for y in range(top, top + height):
                for x in range(left , left + width):
                    if 0 <= x < img.width and 0 <= y < img.height:
                        img.putpixel((x, y), 0)

        # Replace the original pixel_array with the modified one
        new_dicom_instance.PixelData = np.array(img).tobytes()
        #generate a new SOPInstanceUID to follow the DICOM's standard
        new_dicom_instance.SOPInstanceUID = generate_uid()

        # Save the modified DICOM data to a BytesIO buffer
        output_buffer = io.BytesIO()
        new_dicom_instance.save_as(output_buffer)
        output_buffer.seek(0)

		#upload the new dicom instance to Orthanc
        response = orthanc.RestApiPostAfterPlugins('/instances', output_buffer.read())
        
        # decode the byte string into a regular string
        str_response = response.decode('utf-8')

        # parse the JSON data
        json_response = json.loads(str_response)

        if json_response["Status"] == "Success":
            output.AnswerBuffer(json_response['ID'], "text/plain")
    else:
        output.SendMethodNotAllowed("POST")
        
        
      
#set up model  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained = "/python/src/model.pth"
if not os.path.exists(pretrained):
    print("Checkpoint not found")
    

model_name = "SSD"
figsize = 512
trunc = False
backbone_name = "resnet152"

if "SSD" in model_name:
    model = SSD(model_name, trunc, backbone=ResNet(backbone_name), figsize=figsize, num_classes=2)
else:
    model = Textboxes(model_name, trunc, backbone=ResNet(backbone_name), figsize=figsize, num_classes=2)

dboxes = generate_dboxes(model_name, trunc, figsize)
	
encoder = Encoder(dboxes)
transformer = SSDTransformer(dboxes, (figsize, figsize), val=True)

if torch.cuda.is_available():
    checkpoint = torch.load(pretrained)
else:
    checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))

model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint["model_state_dict"].items()}
model.load_state_dict(model_state_dict)
model.to(device) 
model.eval()



# add rest callback
orthanc.RegisterRestCallback('/MedTextCleaner/(.*)', serve_vue_app_files)
orthanc.RegisterRestCallback('/predict', predict)
orthanc.RegisterRestCallback('/redact/(.*)', redact)


# add a "redact text" button in the Orthanc Explorer
with open("/python/extend-explorer.js", "r") as f:
    orthanc.ExtendOrthancExplorer(f.read())


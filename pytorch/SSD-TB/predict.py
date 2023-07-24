import torch
import torchvision
import os
from src.model import Textboxes, ResNet, SSD
from src.dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from src.transform import SSDTransformer
from src.utils import generate_dboxes, Encoder
import csv
import json
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_args():
    parser = ArgumentParser(description="Implementation of SSD/TB")
    
    parser.add_argument("--data-path", type=str, default="coco",
                        help="the root folder of dataset")
    parser.add_argument("--dataset-name", type=str, default="V2",
                        help="the name of dataset")

    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="")

    parser.add_argument("--model", type=str, default="TB_noOffset", choices=["TB_noOffset", "TB", "SSD" "SSD_custom"],
                        help="model name")
    parser.add_argument("--trunc", type=str, default="yes", choices=["yes", "no"],
                        help="Truncation of the model after conv8_2")
    parser.add_argument("--backbone", type=str, default="resnet50", help="resnet[18,34,50,101,152]",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] )
    parser.add_argument("--figsize", type=int, default=300, help="input size, either 300x300 or 512x512",
                        choices=[300,512] )

    parser.add_argument("--nms-threshold", type=float, default=0.5)
    args = parser.parse_args()
    return args

def save_predictions(predictions):
    output_dir = os.path.join("predictions",opt.dataset_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the name of the CSV file you want to create
    filename = os.path.join(output_dir,opt.model_name + ".csv")

    # Define the fieldnames for the first row of the CSV file
    fieldnames = ["image_id", "category_id", "bbox", "score"]

    # Open the CSV file in "write" mode
    with open(filename, mode="w", newline="") as csv_file:

        # Create a writer object using the csv library
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the first row (i.e., the fieldnames)
        writer.writeheader()
        for prediction in predictions:
            int_bbox = []
            for val in prediction["bbox"]:
                int_bbox.append(int(val))
            prediction["bbox"] = int_bbox #reduce memory
            # Write the content of your dictionary to the CSV file
            writer.writerow(prediction)

def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(opt.pretrained):
        print("Checkpoint not found")
        return

    dboxes = generate_dboxes(opt.model, opt.trunc, opt.figsize)
    encoder = Encoder(dboxes)

    if "SSD" in opt.model:
        model = SSD(opt.model, str2bool(opt.trunc), backbone=ResNet(opt.backbone), figsize=opt.figsize, num_classes=2)
    else:
        model = Textboxes(opt.model, str2bool(opt.trunc), backbone=ResNet(opt.backbone), figsize=opt.figsize, num_classes=2)

    transformer = SSDTransformer(dboxes, (opt.figsize,opt.figsize), val=True)

    category_ids = [1]
 
    if torch.cuda.is_available():
        checkpoint = torch.load(opt.pretrained)
    else:
        checkpoint = torch.load(opt.pretrained, map_location=torch.device('cpu'))

    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(model_state_dict)
    model.to(device) 
    model.eval()


    samples = []
    with open(os.path.join(opt.data_path,"annotations", opt.dataset_name + ".json")) as json_file:
        coco = json.load(json_file)

        for image in coco['images']:
            samples.append((image["file_name"],image["id"]))


    predictions = []

    print("processing samples")
    for filepath, img_id in tqdm(samples):
        img = Image.open(os.path.join(opt.data_path,opt.dataset_name,filepath)).convert("RGB")
        width, height = img.size

        img, _, _, _ = transformer(img, None, torch.zeros(1,4), torch.zeros(1))
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            ploc, plabel = model(img.unsqueeze(0))
            ploc, plabel = ploc.float(), plabel.float()
            # scores, candidates = encoder.get_matched_idx(ploc, plabel,opt.nms_threshold , 200)[0]
            # print(scores)
            # print(candidates)
            result = encoder.decode_batch(ploc, plabel,opt.nms_threshold , 200)[0]

            loc, label, prob = [r.cpu().numpy() for r in result]

            for loc_, label_, prob_ in zip(loc, label, prob):
                xmin,ymin, w, h = loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width, (loc_[3] - loc_[1]) * height
                pred = {"image_id": img_id, "category_id": category_ids[label_ - 1], "bbox": [xmin, ymin, w, h], "score": prob_}
                predictions.append(pred)

    save_predictions(predictions)

if __name__ == "__main__":
    opt = get_args()
    main(opt)

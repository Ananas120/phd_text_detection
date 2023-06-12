# SSD: Single Shot MultiBox Detector and TextBoxes

## Introduction

Here is my pytorch implementation of SSD and TextBoxes. These models are based on original model (SSD-VGG16) described in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325) and [TextBoxes](https://arxiv.org/abs/1611.06779).


- **Dataset**:
  Download the coco images and annotations from [onedrive](https://1drv.ms/f/s!Ao0PwLglbpXkhYV3rNSi3q1E1mua-Q?e=cDsahX). Make sure to put the files as the following structure (The root folder names **coco**):
  ```
  coco
  ├── annotations
  │   ├── V2_train.json
  │   └── V2_val.json
  │── V2_train
  └── V2_val 
  ```

## How to use my code


* **Train your model** by running `python train.py --model [SSD|SSD_custom|TB_noOffset|TB] --trunc [yes/no] --backbone [resnet18,resnet34,resnet50,resnet101,resnet152] --figsize [300,512] --batch-size [int] --pretrained path/to/trained_model`. You could stop or resume your training process whenever you want. For example, if you stop your training process after 10 epochs, the next time you run the training script, your training process will continue from epoch 10. 
* **Test your model for V2 val dataset** by running `python predict.py --pretrained path/to/trained_model --model [SSD|SSD_custom|TB_noOffset|TB] --trunc [yes/no] --backbone [resnet18,resnet34,resnet50,resnet101,resnet152] --figsize [300,512] --model-name [str]`

You could download my trained weight for SSD512-Resnet152 and TB_noOffset512-trunc-Resnet152 [link](https://1drv.ms/f/s!Ao0PwLglbpXkhYV3rNSi3q1E1mua-Q?e=cDsahX)


## References
- Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg "SSD: Single Shot MultiBox Detector" [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

- Minghui Liao, Baoguang Shi, Xiang Bai, Xinggang Wang, and Wenyu Liu. "Textboxes: A fast text detector with a single deep neural network", 2016 [TextBoxes](https://arxiv.org/abs/1611.06779).

- My implementation is inspired by and therefore borrows many parts from [NVIDIA Deep Learning examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD) and [ssd pytorch](https://github.com/qfgaohao/pytorch-ssd) and [uvipen SSD-pytorch](https://github.com/uvipen/SSD-pytorch/tree/main)

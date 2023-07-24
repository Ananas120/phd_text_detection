# PhD project on Text Detection in medical images

- **Date** : 2023
- **Author (PhD student)** : Langlois Quentin, UCLouvain, ICTEAM, Belgium.
- **Co-author (Master thesis student)** : Szelagowski Nicolas
- **Supervisor** : Jodogne Sébastien, UCLouvain, ICTEAM, Belgium.

This project is based on the Nicolas' Master thesis project **"Detection and removal of texts burned in medical images using deep learning"**

## Project structure

```bash
├── pytorch     : original code from Nicolas' thesis'
└── tensorflow  : re-implementation of TextBoxes and SSD in tensorflow \*
```

\* This code is based on a copy of [my other PhD project](https://github.com/Ananas120/phd_segmentation) on medical imaging (3D organ segmentation) which is based on [this project](https://github.com/yui-mhcp/detection). The implementation of EAST (evaluated in comparison with TextBoxes and SSD comes from this repo).

## Available models

### Model architectures

Available architectures : TODO

### Model weights

| Library   | Architecture  | Trainer   | Weights   |
| :-------: | :-----------: | :-------: | :-------: |


## Installation and usage

1. Clone this repository : `git clone https://github.com/Ananas120/phd_text_detection.git`
2. Go to the root of this repository : `cd phd_text_detection/{pytorch / tensorflow}`
3. Install requirements : `pip install -r requirements.txt`

## TO-DO list :

- [x] Make the TO-DO list
- [x] Fork the original repo
- [x] Re-organize the repo to have both `pytorch` and `tensorflow` directories
- [ ] Add notes and references
- [ ] Implement `SSD` in `tensorflow`
- [ ] Implement `TextBoxes` in `tensorflow`
- [ ] Copy the available `pytorch` pretrained models to `tensorflow`
- [ ] Copy the pre-processing pipeline in `tensorflow`
- [ ] Copy the post-processing pipeline (i.e. model output decoding) in `tensorflow`
- [ ] Adapt the `predict` method of the  `BaseDetector` class to get the correct format for the evaluation scripts
- [ ] Evaluate the models
    - [ ] TextBoxes
    - [ ] SSD
    - [ ] EAST
- [ ] Add usage examples
- [ ] Add pretrained model weights
- [ ] (Optional) add OCR to filter out false positive

## Contacts and licence

### My contacts

- PhD student email : quentin.langlois@uclouvain.be
- Supervisor email : sebastien.jodogne@uclouvain.be

## Licence

The `tensorflow/` project inherits the [AGPL v3.0](tensorflow/LICENCE) licence (see the `tensorflow/LICENCE` file for more information) from [the original project](https://github.com/yui-mhcp/detection).

The `pytorch/` project contains multiple sub-licences : `MIT` for the model codes and [GPL v3.0](pytorch/OrthencPlugin/LICENCE) for the `OrthencPlugin`. However, the main project does not have any clear licence : all rights reserved to his author (Nicolas).


## Notes and references

Papers : TODO

GitHub projects : 
- [Nicolas' Master Thesis project](https://github.com/NicoSzela/MasterThesis) : the `pytorch/` directory is an exact copy of this repo
- [PhD project on organ segmentation](https://github.com/Ananas120/phd_segmentation) : the `tensorflow/` is highly inspired from this repo (especially for the medical image processing part)
- [The original detection project](https://github.com/yui-mhcp/detection) : original detection project containing the EAST implementation (it is also the base of my PhD project mentionned above).

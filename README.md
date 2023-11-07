# Streetscape and Subway Masked Face Video Dataset (SSMFVD)

## Dataset

Download the SSMFVD dataset from [GoogleDrive](
https://drive.google.com/file/d/1Yk8NnEObDLSOHS-ih1Uowh1YwJNrTE3g/view?usp=sharing), it is generally used for object detection tasks.

Download the cropped SSMFVD dataset from [GoogleDrive](
https://drive.google.com/file/d/1U2-B7IPGbrc3dhCBHyVNvFzaHlA9Iwm5/view?usp=sharing), the cropped dataset is generally used for image classification tasks.

Please note that the dataset provided is for academic research purposes and is not intended for commercial use.


## Code
* The folder `TrainingStage1` contains all the code for the first training stage.
* The folder `TrainingStage2` contains code for the second training stage and some helper functions.

## Environments
Please check the `requirements.txt` file

## Training Stage 1
### File structure
```
  ├── backbone.py: feature extraction network
  ├── transfer_losses.py and loss_funcs: loss functions
  ├── DSAN.yaml: configure the hyper-parameters 
  ├── data_loader.py: used to load and preprocess dataset
  ├── models.py: the transfer learning network framework
  ├── main.ipynb: the main training file for first training stage (Colab)
  └── utils.py: helper functions
```

## Training Stage 2
### File structure:
```
  ├── backbone: feature extraction network
  ├── network_files: Faster R-CNN network
  ├── data: text files contain the image names and paths for training, validation and testing
  ├── my_dataset.py: used to load dataset
  ├── faster_rcnn_baseline.ipynb: faster rcnn with pretrained weights as the baseline model (Colab)
  ├── main.ipynb: the main training file for second training stage (Colab)
  ├── process_weights_from_stage1.ipynb: used to process the weights trained in the first training stage for use in the second training stage
  ├── predict.ipynb: draw or visualize bounding box and test for response time
  ├── evaluate-main.ipynb: evaluate the model performance with trained weights
  └── masked_face_*.json: class to index label files
```

### Download address for pre-trained weights (download and place in the backbone folder):
* ResNet50 imagenet weights backbone: [https://download.pytorch.org/models/resnet50-0676ba61.pth](https://download.pytorch.org/models/resnet50-0676ba61.pth)
* ResNet50+FPN backbone: [https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth)
* Please rename the downloaded backbone weights if necessary (e.g. `resnet50.pth` or `fasterrcnn_resnet50_fpn_coco.pth`)

## How to run
* Please prepare the datasets and pre-trained weights before running the code
* For the first training stage, you can run the `/TrainingStage1/main.ipynb` with Google Colab and make sure configure the directory properly
* Once you obtain the trained weights from first training stage, you can use the `/TrainingStage2/process_weights_from_stage1.ipynb` to process the weights and save the weights under the `/TrainingStage2/backbone/` folder
* Then you can start second training stage with `faster_rcnn_baseline.ipynb`, `main.ipynb`, `evaluate-main.ipynb`, or `predict.ipynb`

## References 
(which relates to code implementation)

[1] Zhu, Yongchun, et al. "Deep subdomain adaptation network for image classification." IEEE transactions on neural networks and learning systems (2020).

[2] https://github.com/jindongwang/transferlearning

[3] https://github.com/pytorch/vision/tree/master/torchvision/models/detection

Full references can be found [here](https://ieeexplore.ieee.org/document/10191466/references#references).

## Citation

If you think this git repo is helpful to you and your research, please cite us!

```
@inproceedings{yang2023ss,
  title={SS-Faster-RCNN: A Domain Adaptation-based Method to Detect Whether People Wear Masks Correctly},
  author={Yang, Boran and Hossain, Md Zakir and Rahman, Shafin},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2023},
  organization={IEEE}
}
```

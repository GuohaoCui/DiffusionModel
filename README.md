# 20241016
## Environment configuration

* Python: 3.9
* Python library
  - OpenCV: 4.7.0.72
  - numpy: 1.23.5
  - scikit-image:0.20.0
  - pytorch:1.12.1 py3.9\_cuda11.6\_cudnn8\_0
  - pyyaml:6.0
  - torchvision:0.5.0
  - dominate:2.4.0
  - visdom:0.1.8.8
## Dataset
Model training requires paired stamp document images and real images, with a dataset structure of:

dataset/
├── train/
│   ├── input/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── target/
│       ├── 1.png
│       ├── 2.png
│       └── ...
├── test/
│   ├── input/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── target/
│       ├── 1.png
│       ├── 2.png
│       └── ...


The model needs to process the image into small blocks of 16 * 16, so the size of the image needs to be a multiple of 16. When the size is not suitable, call the record_image_sizes and resize_images'in_folder methods in train-diffusion. py to handle it. When generating, use the resize_images_from. FILE method in eval-diffusion. py to restore the original size.

## Run
### Training
The model training parameters are configured in the configs.yml file. The key parameters are as follows:
* Training set path: train_data_dir
* Test set path: test_data_dir
* Output path: test_save_dir
* Intermediate result output path: val_save_dir
* Number of training epochs: n_epochs
* Weight path: resume
* Learning rate: lr
* Batch size: batch_size

**model training command**
```
python train_diffusion.py
```
**test**
```
python eval_diffusion.py
```

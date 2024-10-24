
## Environment configuration

* Python: 3.9
* Python library
  - OpenCV: 4.7.0.72
  - numpy: 1.23.5
  - scikit-image: 0.20.0
  - pytorch: 1.12.1 py3.9\_cuda11.6\_cudnn8\_0
  - pyyaml: 6.0
  - torchvision: 0.5.0
  - dominate: 2.4.0
  - visdom: 0.1.8.8
    
## Dataset structure
Model training requires paired stamp document images and real images, with a dataset structure of:

     dataset/
     ├── train/
     │   ├── input/
     │   │   ├── 1.png
     │   │   ├── 2.png
     │   │   └──...
     │   └── target/
     │       ├── 1.png
     │       ├── 2.png
     │       └──...
     ├── test/
     │   ├── input/
     │   │   ├── 1.png
     │   │   ├── 2.png
     │   │   └──...
     │   └── target/
     │       ├── 1.png
     │       ├── 2.png
     │       └──...


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

**train**
```
python train_diffusion.py
```
**test**
```
python eval_diffusion.py
```

## Data
The link to the dataset is [dataset](https://drive.google.com/file/d/1xLqMP_hpbQiuvI_xvblGoR2f7i5y4SdK/view?usp=drive_link), and the link to the pre-trained model is [weight](https://drive.google.com/file/d/1YJepDa6VhTGJfB0Rk9IxkxCbikKZOjhN/view?usp=drive_link). 
In particular, since the complete document image contains sensitive information of many companies, we only provide the stamp area and the corresponding real image. We are trying to find a way to solve this problem. We are very sorry for the trouble caused to you. In addition, since the dataset is randomly divided, the test results may not be exactly the same as in the paper, but generally will not exceed 0.1.

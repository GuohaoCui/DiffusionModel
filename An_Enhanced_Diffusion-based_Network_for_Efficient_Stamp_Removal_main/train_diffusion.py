import argparse
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from dataset import Data
from models import DenoisingDiffusion
from PIL import Image

def config_get():
    parser = argparse.ArgumentParser()
    # 参数配置文件路径
    parser.add_argument("--config", default='./configs.yml', type=str, required=False, help="Path to the config file")
    args = parser.parse_args()

    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def resize_images_in_folder(folder, size=(256, 256)):
    """
    Resize all images in the source_folder to the specified size and save them to the dest_folder.
    
    :param source_folder: The folder containing the original images.
    :param dest_folder: The folder where resized images will be saved.
    :param size: A tuple indicating the new size (width, height) of the images.
    """
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(folder, filename)
            image = Image.open(filepath)
            
            # 调整图像大小
            resized_image = image.resize(size, Image.ANTIALIAS)
            
            # 保存调整大小的图像到目标文件夹
            resized_image.save(os.path.join(folder,filename))

def record_image_sizes(folder, output_file):
    """
    Record the filenames and sizes of images in the specified folder to a text file.

    :param folder: The folder containing the images.
    :param output_file: Path to the text file where image names and sizes will be recorded.
    """
    
    # 打开输出文件准备写入
    with open(output_file, 'w') as file:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(folder, filename)
                # 打开图像并获取其尺寸
                with Image.open(filepath) as img:
                    width, height = img.size

                # 将文件名和图像尺寸写入输出文件
                file.write(f"{filename}: {width}x{height}\n")

    #print(f"Image sizes have been recorded in {output_file}")

def main():
    config = config_get()

    # 判断是否使用 cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("=> using device: {}".format(device))
    config.device = device

    # 随机种子
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    # 加载数据
    DATASET = Data(config)
    _, val_loader = DATASET.get_loaders()

    # 创建模型
    print("=> creating denoising diffusion model")
    diffusion = DenoisingDiffusion(config)
    diffusion.train(DATASET)


if __name__ == "__main__":
# 调用函数：示例路径，你需要根据自己的情况进行修改
    #folder_input = '/public/home/lab70432/CGH_workspace/dataset/train/input/'  # 指定源文件夹路径
    #folder_target = '/public/home/lab70432/CGH_workspace/dataset/train/target/'
    #output_file = '/public/home/lab70432/CGH_workspace/output.txt'
    #record_image_sizes(folder_input, output_file)
    #resize_images_in_folder(folder_input)
    #resize_images_in_folder(folder_target)
    main()

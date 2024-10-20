import argparse
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from dataset import Data
from models import DenoisingDiffusion, DiffusiveRestoration


def config_get():
    parser = argparse.ArgumentParser()
    # 参数配置文件路径
    parser.add_argument("--config", default='/public/home/lab70432/CGH_workspace/diffusion_stamp_main_STN/configs.yml', type=str, required=False, help="Path to the config file")
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
    _, val_loader = DATASET.get_loaders(parse_patches=False)

    # 创建模型
    print("=> creating diffusion model")
    diffusion = DenoisingDiffusion(config)
    model = DiffusiveRestoration(diffusion, config)

    # 恢复图像
    model.restore(val_loader, r=config.data.grid_r)

def resize_images_from_file(folder, input_file):
    """
    Resize images in the specified folder according to the sizes recorded in the input text file.

    :param folder: The folder containing the images to resize.
    :param input_file: The text file with recorded image names and their original sizes.
    """
    
    # 读取输入文件中所有行的信息，并存储在字典中
    size_dict = {}
    with open(input_file, 'r') as file:
        for line in file:
            # 移除换行符，然后分割文件名和尺寸
            parts = line.strip().split(': ')
            filename = parts[0]
            # 假设尺寸格式正确，并分割宽度和高度
            size = parts[1].split('x')
            width, height = int(size[0]), int(size[1])
            size_dict[filename] = (width, height)
    
    # 遍历文件夹中的图像，如果其尺寸信息在字典中，则调整尺寸
    for filename in os.listdir(folder):
        if filename in size_dict:
            filepath = os.path.join(folder, filename)
            new_size = size_dict[filename]
            with Image.open(filepath) as img:
                resized_img = img.resize(new_size, Image.ANTIALIAS)
                resized_img.save(filepath)

    print(f"Images have been resized according to {input_file}")


if __name__ == '__main__':
    main()
    path = '/public/home/lab70432/CGH_workspace/diffusion_stamp_main_STN/results'
    input_file = '/public/home/lab70432/CGH_workspace/output.txt'
    resize_images_from_file(path, input_file)    

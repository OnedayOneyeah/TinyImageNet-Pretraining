
# collect data
import time
# import scipy.ndimage as nd
import imageio as nd
import numpy as np
import random
from pathlib import Path
from robustbench.zenodo_download import DownloadError, zenodo_download

# build model and optimizer
# from resnet50 import ResNet50
from resnet50_ import ResNet50
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# utils
import argparse
from tqdm import tqdm
from collections import OrderedDict
from utils import PCANoisePIL, CustomDataset, collate_fn
import pickle as pkl
import os
import warnings
warnings.filterwarnings('ignore')

# global vars
BATCH_SIZE = 256
# BATCH_SIZE = 128
NUM_CLASSES = 200
EPOCHS = 50
PATH = '/data/tent/data/Tiny-ImageNet-C/'

transform_mean = np.array([ 0.485, 0.456, 0.406 ])
transform_std = np.array([ 0.229, 0.224, 0.225 ])

PREPROCESSINGS = {
    'TinyImageNet':
        transforms.Compose([
            transforms.Resize(64),
            # transforms.RandomResizedCrop(64),
            # transforms.RandomHorizontalFlip(p=0.2),
            # transforms.RandomVerticalFlip(p=0.1),
            
            # PCANoisePIL(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ])
}

CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def load_data(severity: int = 1,
              corruption = CORRUPTIONS[0]
              ):
    n_total_tinyimagenet = 10000 # 50 (img) * 200 (cls) = 10000 
    n_examples = n_total_tinyimagenet
    data_dir = Path(PATH)

    labels_path = data_dir / 'labels.npy'
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path)
    
    
    corruption_file_path = data_dir / (corruption + '.npy')
    
    if not corruption_file_path.is_file():
        raise DownloadError(
            f"{corruption} file is missing, try to re-download it.")
    
    images_all = np.load(corruption_file_path)
    images = images_all[(severity - 1) * n_total_tinyimagenet : severity *n_total_tinyimagenet]
    
    # Make it in the Pytorch format
    x_test = np.transpose(x_test, (0,3,1,2))
    
    x_test = torch.Tensor(x_test)
    trns = transforms.Compose([
        transforms.Resize(64),
        transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                  std = (0.229, 0.224, 0.225))
    ])
        
    x_test = trns(x_test)
    x_test = torch.Tensor(x_test[:n_examples])
    # x_test = x_test[:n_examples]
    y_test = torch.LongTensor(y_test)[:n_examples]
    
    return x_test, y_test

def test_c(model,
           severity: int = 1,
           corruption = CORRUPTIONS[0]):
    x_test, y_test = load_data(severity, corruption)
    
    with torch.no_grad():
       for  
    output = model(x_test)
    

def main(severity: int = 1):
    
    ckpt = torch.load('/home/yewon/energy/ckpt/tinyimagenet/augmix.pth.tar')
    model = ResNet50()
    model.load_state_dict({k.replace("module.", ""): v for k,v in ckpt['state_dict'].items()})
    model.eval()
    
    for corruption in CORRUPTIONS:
        test_c(model, severity, corruption)

if __name__ == "__main__":
    main()
"""
source: https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/ResNet_TinyImageNet.ipynb
"""

# collect data
import time
# import scipy.ndimage as nd
import imageio as nd
import numpy as np
import random

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
PATH = '/data/tent/data/tiny-imagenet-200/'

transform_mean = np.array([ 0.485, 0.456, 0.406 ])
transform_std = np.array([ 0.229, 0.224, 0.225 ])

PREPROCESSINGS = {
    'TinyImageNet':
        transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(64),
            # transforms.RandomHorizontalFlip(p=0.2),
            # transforms.RandomVerticalFlip(p=0.1),
            
            # PCANoisePIL(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ])
}


# # input image dimensions
# img_rows, img_cols = 64, 64
# # The images are RGB
# img_channels = 3

def get_id_dictionary():
    id_dict = {}
    for i, line in enumerate(open( PATH + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
  
def get_class_to_id_dict():
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open( PATH + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result


def get_data(id_dict):
    
    print('starting loading data')
    
    if os.path.exists('./data.pkl'):
        with open('./data.pkl', 'rb') as f:
            data = pkl.load(f)
        
        print( "test data shape: ",  data['X_test'].shape )
        print( "test label shape: ", data['Y_test'].shape )
        # data['X_test'] = data['X_test'].type('torch.FloatTensor')
        
        return data
    
    else:
        test_data, test_labels = [], []
        
        t = time.time()

        for line in open( PATH + 'val/val_annotations.txt'):
            img_name, class_id = line.split('\t')[:2]
            test_data.append(nd.imread( PATH + 'val/images/{}'.format(img_name) ,mode='RGB'))
            test_labels_ = np.array([[0]*200])
            test_labels_[0, id_dict[class_id]] = 1
            test_labels += test_labels_.tolist()

        print('finished loading data, in {} seconds'.format(time.time() - t))
        
        # The data, shuffled and split between train and test sets:
        
        X_test = np.transpose(np.array(test_data), (0,3,1,2))
        Y_test = np.array(test_labels)
        X_test = X_test.astype('float32')
        

        print( "test data shape: ",   X_test.shape )
        print( "test_labels shape: ", Y_test.shape )
        
        return data

from pathlib import Path
def build_model(device:None):
    model = ResNet50()
    # ckpt_dir = "./weights/retraining/tinyimagenet_alpha.pt"
    ckpt_dir = "./weights/retraining/augmix_torch.pt"
    
    checkpoint = torch.load(ckpt_dir)
    
    # load model
    model = ResNet50()
    model.load_state_dict({k.replace("module.", ""): v for k,v in checkpoint['model_state_dict'].items()})
    # model.load_state_dict({k.replace("module.", ""): v for k,v in checkpoint['state_dict'].items()})
    return model

def get_argparser():
    # parse arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization",
                        default = 'he_normal', # random_uniform, random_normal, he_normal
                        type = str,
                        help = "model initialization method")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")        
    parser.add_argument("--num_workers", type=int, default=2,
                        help="number of workers on dataloaders")
    
    return parser

def test(model, test_loader):
    model.eval()
    correct = 0 
    
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            preds = model(image)
            prediction = preds.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            
    test_acc = 100. * correct / len(test_loader.dataset)
    
    return test_acc
    
    
if __name__ == "__main__":
    cfg = get_argparser().parse_args()
    print("Collecting data...")
    print("===============")
    
    data = get_data(get_id_dictionary())
    
    print("===============")
    print("Data Collected")
    
    test_dataset = CustomDataset(data['X_test'], data['Y_test'], transform = PREPROCESSINGS['TinyImageNet'])
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = BATCH_SIZE,
                                              shuffle = False,
                                              collate_fn = collate_fn, num_workers=cfg.num_workers
                                              )
    
    print("Building Renset 50...")
    print("=====================")
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: %s" % DEVICE)
    model = build_model(DEVICE)
    model = nn.DataParallel(model).to(DEVICE)
    
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    print("=====================")
    print("Done.")
    
    test_acc = test(model, test_loader)
    print("Test Acc. {:.2f}".format(test_acc))
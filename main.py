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
import torch.backends.cudnn as cudnn

# utils
import argparse
from tqdm import tqdm
from collections import OrderedDict
from utils import PCANoisePIL, CustomDataset, AugMixDataset, collate_fn
import pickle as pkl
import os
import torch.nn.functional as F
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
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.1),
            
            # PCANoisePIL(),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ]),
        
    'Augmix':
        transforms.Compose([
            transforms.Resize(64),
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor()
            
        ]),
    'Normalize':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ]),
    'TinyImageNet_alpha':
        [
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomAutocontrast(), # default p = 0.5
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomAffine(degrees = (30,70), translate = (0.1, 0.3), scale = (0.5, 0.75)), # replacing shear_x, shear_y
            
            # transforms.RandomSolarize(threshold=192.0),
            transforms.Grayscale(num_output_channels = 3)
            # transforms.RandomEqualize(),
            # transforms.RandomPosterize(bits = 2),
            
            # transforms.Normalize(mean = transform_mean, std = transform_std)
        ],
        
    'Augmix_torch':
            transforms.Compose([
            transforms.AugMix(all_ops = False), # imagenet-c overlapping corruptions are excluded
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std)
        ])
        
}


# # input image dimensions
# img_rows, img_cols = 64, 64
# # The images are RGB
# img_channels = 3

def get_id_dictionary():
    id_dict = OrderedDict()
    for i, line in enumerate(open( PATH + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
        
    with open('./id_dict.pkl','wb') as f:
        pkl.dump(id_dict, f)
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
        
    # with open('./result.pkl','wb') as f:
    #     pkl.dump(id_dict, f)
    return result

def get_data(id_dict):
    
    print('starting loading data')
    
    if os.path.exists('./data.pkl'):
        with open('./data.pkl', 'rb') as f:
            data = pkl.load(f)
        
        print( "train data shape: ",  data['X_train'].shape )
        print( "train label shape: ", data['Y_train'].shape )
        print( "test data shape: ",   data['X_test'].shape )
        print( "test_labels shape: ", data['Y_test'].shape )
        return data
    else:
        train_data, test_data = [], []
        train_labels, test_labels = [], []
        t = time.time()
        for key, value in id_dict.items():
            train_data += [nd.imread( PATH + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), mode='RGB') for i in range(500)]
            train_labels_ = np.array([[0]*200]*500)
            train_labels_[:, value] = 1
            train_labels += train_labels_.tolist()

        for line in open( PATH + 'val/val_annotations.txt'):
            img_name, class_id = line.split('\t')[:2]
            test_data.append(nd.imread( PATH + 'val/images/{}'.format(img_name) ,mode='RGB'))
            test_labels_ = np.array([[0]*200])
            test_labels_[0, id_dict[class_id]] = 1
            test_labels += test_labels_.tolist()

        print('finished loading data, in {} seconds'.format(time.time() - t))
        
        # The data, shuffled and split between train and test sets:
        
        X_train = np.array(train_data)
        X_test = np.array(test_data)
        Y_train = np.array(train_labels)
        Y_test = np.array(test_labels)
        
        # X_train = np.transpose(np.array(train_data), (0,3,1,2)) # (b,c,h,w)
        # X_test = np.transpose(np.array(test_data), (0,3,1,2)) # (b,c,h,w)
        

        # X_train = X_train.astype('float32')
        # X_test = X_test.astype('float32')
        
        print( "train data shape: ",  X_train.shape )
        print( "train label shape: ", Y_train.shape )
        print( "test data shape: ",   X_test.shape )
        print( "test_labels shape: ", Y_test.shape )
        
        data = OrderedDict({'X_train': X_train,
                'Y_train': Y_train,
                'X_test': X_test,
                'Y_test': Y_test})
        
        with open('./data.pkl', 'wb') as f:
            pkl.dump(data, f)
        return data
  
def build_model_and_optimizer(device:None):
    
    model = ResNet50()
    model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr = 0.01)
    optimizer = optim.SGD(model.parameters(), lr = 0.01, weight_decay = 0.001, momentum = 0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS//10)

    return model, optimizer, scheduler


def train(model, 
          train_loader,
          criterion, 
          scheduler,
          augmix = False):
    
    # setting    
    model.train()
    train_loss = 0.
    correct = 0
    tqdm_bar = tqdm(train_loader)
    
    for image, label in tqdm_bar:
        # print(len(image)) # 3
        # print(image[0].shape) # (batch_size, c, h, w)
        optimizer.zero_grad()
        
        if augmix:
            images_all, label = torch.cat(image, 0).cuda(), label.cuda()
            logits_all = model(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, image[0].size(0)) # split into three parts
            
            loss = criterion(logits_clean, label)
            
            p_clean, p_aug1, p_aug2 = F.softmax(
                    logits_clean, dim=1), F.softmax(
                        logits_aug1, dim=1), F.softmax(
                            logits_aug2, dim=1)
            
            # clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2)/3., 1e-7, 1)
            p_mixture = torch.log(p_mixture+1e-7)
            
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            
            prediction = (logits_clean + logits_aug1 + logits_aug2).max(1, keepdim = True)[1]
            
        else: # no augmix, no_jsd both applies
            
            image, label = image.to(DEVICE), label.to(DEVICE)
            preds = model(image)
            loss = criterion(preds, label)
            prediction = preds.max(1, keepdim = True)[1]
        
        correct += prediction.eq(label.view_as(prediction)).sum().item()    
        print(loss.item())
        loss.backward()
        train_loss += float(loss.item())
        
        optimizer.step()
        tqdm_bar.set_description("Epoch {} - train loss {:.6f}".format(epoch, loss.item()))
    
    scheduler.step()
    # make dataloader
    
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)
    return train_loss, train_acc

def evaluate(model,
             test_loader,
             criterion,
            ):
    # setting
    model.eval()
    test_loss = 0.
    correct = 0
    
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            preds = model(image)
            test_loss += criterion(preds, label).item()
            prediction = preds.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    return test_loss, test_acc

def get_argparser():
    # parse arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization",
                        default = 'he_normal', # random_uniform, random_normal, he_normal
                        type = str,
                        help = "model initialization method")
    parser.add_argument("--augmix",
                        default = False,
                        action='store_true',
                        help = "apply augmix")
    parser.add_argument("--no_jsd",
                        default = False,
                        action = 'store_true',
                        help =  'w.o. KL div')
    parser.add_argument("--alpha_trans",
                        default = False,
                        action = 'store_true',
                        help = 'apply additional augmentations')
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")        
    parser.add_argument("--num_workers", type=int, default=2,
                        help="number of workers on dataloaders")
    
    return parser

if __name__ == "__main__":
    
    cfg = get_argparser().parse_args()
    print("Collecting data...")
    print("===============")
    
    data = get_data(get_id_dictionary())
    
    print("===============")
    print("Data Collected")
    
    if cfg.augmix:
        train_dataset = AugMixDataset(PREPROCESSINGS['Augmix'](data['X_train']), data['Y_train'], preprocess = PREPROCESSINGS['Normalize'])
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                               batch_size = BATCH_SIZE,
                                               shuffle = True,
                                               num_workers = cfg.num_workers)
    else:
        if cfg.no_jsd:
            train_dataset = AugMixDataset(PREPROCESSINGS['Augmix'](data['X_train']), data['Y_train'], preprocess = PREPROCESSINGS['Normalize'], no_jsd = True)
        else:
            if cfg.alpha_trans:
                train_dataset = CustomDataset(data['X_train'], data['Y_train'], transform = PREPROCESSINGS['TinyImageNet_alpha'], normalization = PREPROCESSINGS['Normalize'])
            else:
                # train_dataset = CustomDataset(data['X_train'], data['Y_train'], transform = PREPROCESSINGS['TinyImageNet'])
                train_dataset = CustomDataset(data['X_train'], data['Y_train'], transform = PREPROCESSINGS['Augmix_torch'])
                
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                               batch_size = BATCH_SIZE,
                                               shuffle = True,
                                               collate_fn = collate_fn, num_workers=cfg.num_workers)
    test_dataset = CustomDataset(data['X_test'], data['Y_test'], transform = PREPROCESSINGS['Normalize'])
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = BATCH_SIZE,
                                              shuffle = False,
                                              collate_fn = collate_fn, num_workers=cfg.num_workers
                                              )
    print("Dataset: %s, Train set: %d, Val set: %d" %
          ("TinyImageNet", len(train_loader), len(test_loader)))
    print(f"Type check ... {type(train_dataset.x_data[0][0,0,0])}")

    # build model, optimizer, and scheduler
    print("Building Renset 50...")
    print("=====================")
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: %s" % DEVICE)
    model, optimizer, scheduler = build_model_and_optimizer(DEVICE)
    model = nn.DataParallel(model).to(DEVICE)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    # Setup random seed
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    print("=====================")
    print("Done.")
    
    # train
    print("Train starts~!!!")
    print(f"OPTIONS: augmix: {cfg.augmix}, no_jsd: {cfg.no_jsd}, alpha_aug: {cfg.alpha_trans}")

    best_acc = 0.
    for epoch in range(1, EPOCHS + 1):

        # model = nn.DataParallel(model)
        train_loss, train_accuracy = train(model, train_loader, criterion, scheduler, cfg.augmix)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        print("\n[EPOCH: {}], \tLR: {:.4f}, \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.2f} %, \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, scheduler.get_last_lr()[-1], train_loss, train_accuracy, test_loss, test_accuracy))

        MODEL_PATH = f'./weights/retraining_tmax:5'
        
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        
        if test_accuracy >= best_acc:
            print("New checkpoint! Test Acc.", test_accuracy)
            torch.save({
                'epoch' : epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'test_acc': test_accuracy
            },
                MODEL_PATH + '/augmix_torch.pt'
                    )
            # torch.save(model.module.state_dict(), MODEL_PATH + '/best-model.pt')
            best_acc = test_accuracy
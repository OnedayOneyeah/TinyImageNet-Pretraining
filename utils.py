# import modules
import numpy as np
import random
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
from augmix import aug

class PCANoisePIL(object):
    def __init__(self,
                 alphastd=0.1,
                 eigval=np.array([55.46, 4.794, 1.148]),
                 eigvec=np.array([[-0.5675, 0.7192, 0.4009],
                                  [-0.5808, -0.0045, -0.8140],
                                  [-0.5836, -0.6948, 0.4203],])
                 ):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec
        self.set_alpha()

    def __call__(self, img):

        # 1. pil to numpy
        img_np = np.array(img)                                   # [H, W, C]
        offset = np.dot(self.eigvec * self.alpha, self.eigval)
        img_np = img_np + offset
        img_np = np.maximum(np.minimum(img_np, 255.0), 0.0)
        ret = Image.fromarray(np.uint8(img_np))
        return ret

    def set_alpha(self, ):
        # change per each epoch
        self.alpha = np.random.normal(0, self.alphastd, size=(3,))
        
        
"""
Customized dataset for tiny-imagenet-200

"""
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, transform = None, normalization = None):  # transform is a list if alpha_trans option is on.

        self.x_data = x_data
        self.y_data = torch.LongTensor(y_data.argmax(axis=1))
        # print(self.y_data.shape, self.y_data.max(0).values)
        self.len = self.y_data.shape[0]
        
        self.transform = transform
        self.normalization = normalization
        # self.transform = transform

    def __len__(self):
        """
        Returns the length of self.data.
        """

        return self.len

    def __getitem__(self, idx):
        """
        Returns data point with the given index.
        """
        
        # if self.transform: # augmentation: tinyimagenet / tinyimagenet_alpha
        if type(self.transform) == list:
          sample_size = random.randint(0,len(self.transform)-1)
          trns = transforms.Compose([transforms.Resize(64)]+random.sample(self.transform, k=sample_size))
          
          # nd array to pil image / shape: (64,64,3)
          return self.normalization(trns(Image.fromarray(self.x_data[idx]))), self.y_data[idx] # totensor, noramlization included
            
        else: # tinyimagenet, test_data, augmix_torch (totensor, normalization included)
          return self.transform(Image.fromarray(self.x_data[idx])), self.y_data[idx]
          
"""
(3) Implement the function collate_fn.
"""

def collate_fn(data_samples):
    """Takes in the input samples from the Dataset, and makes them into a batch for the DataLoader.

    - Convert the PIL images in data_samples to pytorch tensors using ToTensor().
    - Convert the labels in data_samples to pytorch tensors.
      (hint : You should put the label in a list before transforming it into a tensor)
    - Append preprocessed image and label tensors to batch_x and batch_y respectively.
    - Convert the batch of image tensors into PyTorch float tensors.
    - Convert the batch of labels into PyTorch long tensors.
      (hint : Refer to the shapes of the Returns below)

    Args:
      data_samples: list of tuples, each containing a PIL image and an integer label

    Returns:
      batch_x: batch of image tensors. size: (BATCH, CHANNEL, HEIGHT, WIDTH)
      batch_y: batch of label tensors. size: (BATCH)
    """
    batch_x, batch_y = [], []
    
    for x, y in data_samples:
      batch_x.append(x)
    #   print(y)
      batch_y.append(int(y.item()))
      

    batch_x = torch.stack(batch_x, dim = 0) # list of tensors to a tensor
    batch_y = torch.LongTensor(batch_y)
    
    return (batch_x, batch_y)
  
class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, x_data, y_data, preprocess, no_jsd=False):
    self.x_data = x_data
    self.y_data = torch.argmax(y_data, dim = 1)
    
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    
    x, y = self.x_data[i], self.y_data[i]
    
    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      
      return im_tuple, y

  def __len__(self):
    return len(self.y_data)


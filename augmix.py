import augmentations
import numpy as np
import torch
import copy
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms

def apply_op(image, op, severity):
    pil_image = to_pil_image(image)
    pil_image = op(pil_image, severity)
    return transforms.ToTensor()(pil_image)

def aug(image, preprocess):
    """
  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.

  Returns:
    mixed: Augmented and mixed image.
  """
    
    aug_list = augmentations.augmentations
    
    # parameters
    mixture_width = 3
    mixture_depth = -1
    aug_prob_coeff = 1.
    aug_severity = 2
    
    ws = np.float32(
        np.random.dirichlet([aug_prob_coeff]*mixture_width)
    )
    m = np.float32(
        np.random.beta(aug_prob_coeff, aug_prob_coeff)
    )
    
    mix = torch.zeros_like(image)
    
    for i in range(mixture_width):
        # image_aug = copy.deepcopy(image) # tensor
        image_aug = image.detach().clone()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(1,4)
        
        for _ in range(depth):
            op = np.random.choice(aug_list)
            # image_aug = op(image_aug, aug_severity)
            image_aug = apply_op(image_aug, op, aug_severity)
            
        mix += ws[i] * preprocess(image_aug)
    
    mixed = (1-m) * (preprocess(image)) + m * mix
    # print(mixed.size()) # (3,64,64)
    return mixed
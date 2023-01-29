import torch
import torch.nn.functional as F
import numpy as np
from math import sqrt, ceil
from torchvision import transforms
from PIL import Image

def resize_322(img: Image.Image):
    resize = transforms.Compose([
        transforms.Resize((322, 322))
    ])

    return resize(img)
    
def center_crop_224(img:Image.Image):
    center_crop = transforms.Compose([
        transforms.CenterCrop((224, 224))
    ])
    return center_crop(img)

def resize_center_crop(img:Image.Image):
    center_crop = transforms.Compose([
        transforms.Resize(256 * np.sqrt(2)),
        transforms.CenterCrop(224)
    ])

    return center_crop(img)

def imagenet_normalize(img:torch.Tensor):
    normalize = transforms.Compose([
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    return normalize(img)

def tensorize(img:Image.Image):
    tensorize = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return tensorize(img)



def get_spatial_transform():
    transform_config = {
        'flip_horizon' : int(torch.rand(1) > 0.5),
        'flip_vertical' :  int(torch.rand(1) > 0.5),
        'rot_angle' : (180 * torch.rand(1)).item(),
        # 'scale' : (0.8 - 1.2) * torch.rand(1) + 1.2,
    }

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(transform_config['flip_horizon']),
        transforms.RandomVerticalFlip(transform_config['flip_vertical']),
        transforms.RandomRotation((transform_config['rot_angle'], transform_config['rot_angle'])),
    ])

    inv_transform = transforms.Compose([
        transforms.RandomRotation((-transform_config['rot_angle'], -transform_config['rot_angle'])),
        transforms.RandomVerticalFlip(transform_config['flip_vertical']),
        transforms.RandomHorizontalFlip(transform_config['flip_horizon']),
        
    ])

    return transform, inv_transform

def get_color_transform():
    transform = transforms.Compose([
        transforms.ColorJitter(
            brightness = 0.5, 
            hue = 0.5,
            contrast = (0, 3),
            saturation = (0, 3)
        ),
    ])

    return transform


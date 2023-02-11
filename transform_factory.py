import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from math import sqrt, ceil
from torchvision import transforms
from PIL import Image
import random

from torchvision.transforms import InterpolationMode

def add_noise(img: Image.Image, mean, sd):
    return min(max(0, img + random.normalvariate(mean, sd)), 255)


def resize_232(img: Image.Image):
    resize = transforms.Compose([
        transforms.Resize((232, 232))
    ])

    return resize(img)

def resize_322(img: Image.Image):
    resize = transforms.Compose([
        transforms.Resize((322, 322), InterpolationMode.BICUBIC)
    ])

    return resize(img)

def resize_224(img):
    resize = transforms.Compose([
        transforms.Resize((224, 224), InterpolationMode.BICUBIC)
    ])
    return resize(img)
    
def center_crop_224(img):
    center_crop = transforms.Compose([
        transforms.CenterCrop((224, 224))
    ])
    return center_crop(img)

def center_crop_11(img:Image.Image):
    center_crop = transforms.Compose([
        transforms.CenterCrop((11, 11))
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

def PIL2Tensor(img:Image.Image):
    t = transforms.Compose([
        transforms.PILToTensor(),
    ])

    return t(img)

def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    # sigma = 1
    sigma = 0.5
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

def ToPIL(img):
    t = transforms.Compose([
        transforms.ToPILImage(),
    ])
    
    return t(img)


def get_spatial_transform():
    transform_config = {
        'flip_horizon' : int(torch.rand(1) > 0.5),
        'flip_vertical' :  int(torch.rand(1) > 0.5),
        # 'rot_angle' : (-90 * torch.rand(1) + 45).item(),
        'rot_angle' : (-360 * torch.rand(1) + 180).item(),

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

    return transform, inv_transform, transform_config

def get_color_transform():
    transform = transforms.Compose([
        transforms.ColorJitter(
            brightness = 0.15, 
            hue = 0.15,
            contrast = 0.25,
            saturation = 0.25
        ),
    ])

    return transform


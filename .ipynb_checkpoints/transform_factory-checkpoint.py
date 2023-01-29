import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

def center_crop(img:Image.Image):
    center_crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    return center_crop(img)

def imagenet_normalize(img:Image.Image):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    
    return normalize(img)



def get_tta_transform():
    transform_config = {
        'flip_horizon' : int(torch.rand(1) > 0.5),
        'flip_vertical' :  int(torch.rand(1) > 0.5),
        'rot_angle' : 180 * torch.rand(1),
        'scale' : (0.8 - 1.2) * torch.rand(1) + 1.2,
    }
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(transform_config['flip_horizon']),
        transforms.RandomVerticalFlip(transform_config['flip_vertical']),
        transforms.RandomRotation((transform_config['rot_angle'], transform_config['rot_angle'])),
        transforms.ToTensor()
    ])

    inv_transform = transforms.Compose([
        transforms.RandomRotation((-transform_config['rot_angle'], -transform_config['rot_angle'])),
        transforms.RandomVerticalFlip(transform_config['flip_vertical']),
        transforms.RandomHorizontalFlip(transform_config['flip_horizon'])
    ])

    return transform, inv_transform
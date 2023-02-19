import torch
# import torch.nn.functional as F
import numpy as np
from torch import Tensor
from math import sqrt, ceil
from torchvision import transforms
from PIL import Image
import random

from torchvision.transforms import InterpolationMode
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Variable
import torchvision.transforms.functional as F


def add_noise(img: Image.Image, mean, sd):
    return min(max(0, img + random.normalvariate(mean, sd)), 255)

def center_crop_32(img: Image.Image):
    center_crop = transforms.Compose([
        transforms.CenterCrop((32, 32))
    ])

    return center_crop(img)

def resize_46(img:Image.Image):
    resize = transforms.Compose([
        transforms.Resize((46, 46), InterpolationMode.BICUBIC)
    ])

    return resize(img)

def resize_32(img:Image.Image):
    resize = transforms.Compose([
        transforms.Resize((32, 32), InterpolationMode.BICUBIC)
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
    transform_space = {
        "Rotate": (torch.linspace(0.0, 135.0, 31), True),
    }


    magnitudes, signed = transform_space['Rotate']

    rot_angle = (
        float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
        if magnitudes.ndim > 0
        else 0.0
    )
    if signed and torch.randint(2, (1,)):
        rot_angle *= -1.0

    if int(torch.rand(1) > 0.5):
        rot_angle = 0

    transform_config = {
        'flip_horizon' : int(torch.rand(1) > 0.5),
        'rot_angle' : rot_angle,

    }

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(transform_config['flip_horizon']),
        # transforms.RandomVerticalFlip(transform_config['flip_vertical']),
        transforms.RandomRotation((transform_config['rot_angle'], transform_config['rot_angle']), InterpolationMode.BILINEAR),
    ])

    inv_transform = transforms.Compose([
        transforms.RandomRotation((-transform_config['rot_angle'], -transform_config['rot_angle']), InterpolationMode.BILINEAR),
        # transforms.RandomVerticalFlip(transform_config['flip_vertical']),
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

def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


class TrivialAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        logger,
        hflip,
        config = None,
        trans_opt='all',
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.logger = logger
        self.hflip = hflip
        self.config = config
        self.trans_opt = trans_opt

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:

        if self.trans_opt == 'all':
            return {
                # op_name: (magnitudes, signed)
                "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
                "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
                "Color": (torch.linspace(0.0, 0.99, num_bins), True),
                "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
                "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
                "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
                "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
                "AutoContrast": (torch.tensor(0.0), False),
                "Equalize": (torch.tensor(0.0), False),
            }
        elif self.trans_opt == 'color':
            return {
                # op_name: (magnitudes, signed)
                "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
                "Color": (torch.linspace(0.0, 0.99, num_bins), True),
                "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
                "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
                "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
                "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
                "AutoContrast": (torch.tensor(0.0), False),
                "Equalize": (torch.tensor(0.0), False),
            }
        elif self.trans_opt == 'spatial':
            return {
                # op_name: (magnitudes, signed)
                "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)


        if self.config is None:
            config = [('hflip', self.hflip)]

            for i in range(len(op_meta) - 1):
                if torch.rand(1) > 0.5:
                    op_index = i
                    op_name = list(op_meta.keys())[op_index]
                    magnitudes, signed = op_meta[op_name]
                    magnitude = (
                        float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
                        if magnitudes.ndim > 0
                        else 0.0
                    )
                    if signed and torch.randint(2, (1,)):
                        magnitude *= -1.0

                    img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                    config.append((op_name, magnitude))

                else: continue
            
            if torch.rand(1) > 0.5:
                op_index = len(op_meta) - 1
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                magnitude = (
                    float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
                    if magnitudes.ndim > 0
                    else 0.0
                )
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0

                img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                config.append((op_name, magnitude))

            if self.logger is not None:
                self.logger.save_transform_config(config)   

            if len(config) == 1:
                return _apply_op(img, "Identity", 0.0, interpolation=self.interpolation, fill = fill)
            else:
                return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # Convert the image to a tensor
        img_tensor = transforms.ToTensor()(img)

        # Generate the noise tensor
        noise = torch.randn(img_tensor.size()) * self.std + self.mean

        # Add the noise to the image tensor
        img_tensor += noise

        # Clamp the image tensor to the range [0, 1]
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Convert the image tensor back to an image
        img = transforms.ToPILImage()(img_tensor)

        return img


def get_trivial_augment(logger=None, aopc=False, trans_opt = 'all', noise_std = 0.05):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if aopc == False:
        transform_config = {
            'flip_horizon': int(torch.rand(1) > 0.5)
        }

        trans = [
            AddGaussianNoise(mean=0.0, std=noise_std),
            transforms.RandomHorizontalFlip(transform_config['flip_horizon']),
            ]
        operator = TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC, logger=logger, hflip=transform_config['flip_horizon'])
        trans.append(operator)
        trans.extend([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean = mean, std = std),

        ])
    else:
        if trans_opt == 'color':
            trans = [
                AddGaussianNoise(mean=0.0, std=noise_std)
            ]

            operator = TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC, logger=logger, hflip=None, trans_opt=trans_opt)
            trans.append(operator)
        
            trans.extend([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean = mean, std = std),

            ])
        elif trans_opt == 'spatial':
            transform_config = {
                'flip_horizon': int(torch.rand(1) > 0.5)
            }
            trans = [
                transforms.RandomHorizontalFlip(transform_config['flip_horizon'])
            ]
            operator = TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR, logger=logger, hflip=transform_config['flip_horizon'], trans_opt=trans_opt)
            trans.append(operator)

        return transforms.Compose(trans)


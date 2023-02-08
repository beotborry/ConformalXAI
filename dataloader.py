import torch
import numpy as np
from PIL import Image
from arguments import get_args
from conformalize import ConformalExpl
from expl import ExplFactory
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights, resnet18, ResNet18_Weights
from utils import set_seed
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from transform_factory import resize_322, center_crop_224, tensorize, get_spatial_transform, get_color_transform, imagenet_normalize
from logger import Logger
from tqdm import tqdm


from os import listdir
from os.path import isfile, join
import random
from utils import set_seed
import numpy as np

from torch.nn.functional import softmax
if __name__ == "__main__":
    base_img_path = "/home/juhyeon/Imagenet/train"
    seed = 2
    set_seed(seed)


    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model = resnet50(weights = ResNet50_Weights.DEFAULT).eval().cuda()


    class_list = listdir(base_img_path)
    class_list = sorted(class_list)
    # print(class_list, len(class_list))
    img_path_list = []

    for label, c in tqdm(enumerate(class_list)):
        filelist = listdir(join(base_img_path, c))
        filelist = random.choices(filelist, k = 1000)
        for f in filelist:
            img_path = join(base_img_path, c, f)
            img = Image.open(img_path)

            # print(np.array(img).shape)
            if len(np.array(img).shape) != 3 or np.array(img).shape[2] != 3:
                continue
            img = imagenet_normalize(tensorize(resize_322(img))).cuda()

            pred = model(img.unsqueeze(0))

            if pred.argmax() == label and softmax(pred).squeeze().max() >= 0.05:
                img_path_list.append(img_path)
                break
            else:
                continue


    with open(f"./train_seed_{seed}.npy", "wb") as f:
        np.save(f, img_path_list)

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
from transform_factory import resize_232, center_crop_224, tensorize, get_spatial_transform, get_color_transform, imagenet_normalize, resize_322
from logger import Logger
from tqdm import tqdm


from os import listdir
from os.path import isfile, join
import random
from utils import set_seed
import numpy as np
from argparse import ArgumentParser

from torch.nn.functional import softmax
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    seed = args.seed   
    base_img_path = "/home/juhyeon/Imagenet/val"
    set_seed(seed)


    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model = resnet50(weights = ResNet50_Weights.DEFAULT).eval().cuda()
    # model = resnet18(weights = ResNet18_Weights.DEFAULT).eval().cuda()
    # model = resnet34(weights = ResNet34_Weights.DEFAULT).eval().cuda()


    class_list = listdir(base_img_path)
    class_list = sorted(class_list)
    class_list = np.array(class_list)

    class_idx = np.arange(0, 1000, 1)
    # np.random.shuffle(class_idx)
    # class_idx = class_idx[:100]

    # print(class_list, len(class_list))
    img_path_list = []

    prob_list = []
    total_data = 0
    for label, c in tqdm(zip(class_idx, class_list)):
        n_selected = 0
        filelist = listdir(join(base_img_path, c))

        np.random.shuffle(filelist)
        for f in filelist:
            img_path = join(base_img_path, c, f)
            img = Image.open(img_path)

            # print(np.array(img).shape)
            if len(np.array(img).shape) != 3 or np.array(img).shape[2] != 3:
                continue

            total_data += 1
            img = imagenet_normalize(tensorize(center_crop_224(resize_322(img)))).cuda()
            # img = imagenet_normalize(tensorize(resize_322(img))).cuda()

            pred = model(img.unsqueeze(0))


            if pred.argmax() == label:
                prob_list.append((img_path, softmax(pred).squeeze().max().item()))
            else:
                continue
        print(len(prob_list))


    with open(f"./val_prob_322_then_224_once.npy", "wb") as f:
        np.save(f, np.stack(prob_list))

    print(total_data)

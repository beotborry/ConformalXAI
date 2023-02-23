import torch
import numpy as np
from PIL import Image
from arguments import get_args
from conformalize import ConformalExpl
from expl import ExplFactory
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights, resnet18, ResNet18_Weights
from utils import set_seed
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
import torch.nn as nn
import os
import time
import torchvision.datasets as datasets
from resnet import resnet20
import torch.distributed as dist

if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    set_seed(args.seed)
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # dist.init_process_group("gloo", rank=0, world_size=2)

    # device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device)

    if args.model == 'resnet50':
        # device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        # torch.cuda.set_device(device)
        # device = torch.device(f"cuda:{args.device[0]},{args.device[1]}" if torch.cuda.is_available() else 'cpu')
        model = resnet50(weights = ResNet50_Weights.DEFAULT).eval().cuda()
        model = DataParallel(model)
    
    elif args.model == 'resnet20':
        device_ids = [args.device, args.device + 1]
        model = resnet20()
        check_point = torch.load('./pretrained_models/resnet20-12fca82f.th', map_location='cuda:%d' % device_ids[0])

        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(check_point['state_dict'])
        model.cuda()

    if args.data == 'imagenet':
        if args.img_path is None:
            with open(f"{args.split}_{args.dataset}_seed_{args.seed}.npy", "rb") as f:
                img_path_list = np.load(f, allow_pickle=True)

        else:
            img_path_list = []
            img_path_list.append(args.img_path)


        for img_path in tqdm(img_path_list):
            orig_img = Image.open(img_path)
            orig_img = orig_img.convert("RGB")
            expl_func = ExplFactory().get_explainer(model = model, expl_method = args.expl_method, upsample=args.upsample)
            conformalizer = ConformalExpl(orig_img, expl_func, args, img_path=img_path)
            
#             if args.with_config:
#                 with open(f"{conformalizer.logger.save_path}/{conformalizer.logger.base_logname}_transform_config.txt", "r") as f:
#                     transform_configs = f.readlines()
                        
            if os.path.exists(f"{conformalizer.logger.save_path}/{conformalizer.logger.base_logname}_orig_true_config.npy"):
                continue
            if args.run_option == 'all':
                conformalizer.make_confidence_set()
                conformalizer.evaluate()
            elif args.run_option == 'eval':
                conformalizer.evaluate()
            elif args.run_option == "pred" or args.run_option == "test":
                conformalizer.make_confidence_set()
    elif args.data == 'cifar10':
        val_loader = torch.utils.data.DataLoader(
            datasets.CiIFAR10(root='./data', train=False), batch_size=1, shuffle=False, pin_memory=True
        )

        dataset = val_loader.dataset
        correct_indicies = torch.load("./cifar10_val_correct_indicies.pt")

        for idx in correct_indicies:
            img_path = f"./data/cifar-10-batches-py/test_{idx}"
            orig_img = dataset[idx][0]
            expl_func = ExplFactory().get_explainer(model = model, expl_method=args.expl_method, upsample=args.upsample)
            conformalizer = ConformalExpl(orig_img, expl_func, args, img_path=img_path)

            if os.path.exists(f"{conformalizer.logger.save_path}/{conformalizer.logger.base_logname}_orig_true_config.npy"):
                continue
            if args.run_option == 'all':
                conformalizer.make_confidence_set()
                conformalizer.evaluate()
            elif args.run_option == 'eval':
                conformalizer.evaluate()
            elif args.run_option == "pred" or args.run_option == "test":
                conformalizer.make_confidence_set()

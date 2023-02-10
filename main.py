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
import torch.nn as nn

if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model = resnet50(weights = ResNet50_Weights.DEFAULT).eval().cuda()
    # model = resnet18(weights = ResNet18_Weights.DEFAULT).eval().cuda()
    # model = resnet34(weights = ResNet34_Weights.DEFAULT).eval().cuda()

    if args.img_path is None:
        with open(f"{args.split}_seed_{args.seed}.npy", "rb") as f:
            img_path_list = np.load(f)

    else:
        img_path_list = []
        img_path_list.append(args.img_path)


    for img_path in tqdm(img_path_list):
        orig_img = Image.open(img_path)
        expl_func = ExplFactory().get_explainer(model = model, expl_method = args.expl_method, upsample=args.upsample)
        conformalizer = ConformalExpl(orig_img, expl_func, args, img_path=img_path)
                

        if os.path.exists(f"{conformalizer.logger.save_path}/{conformalizer.logger.base_logname}_orig_true_config.npy"):
            continue
        if args.run_option == 'all':
            conformalizer.make_confidence_set()
            conformalizer.evaluate()
        elif args.run_option == 'eval':
            conformalizer.evaluate()
        elif args.run_option == "pred" or args.run_option == "get_transform":
            conformalizer.make_confidence_set()
        # conformalizer.logging()


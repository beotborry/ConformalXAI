import torch
from PIL import Image
from arguments import get_args
from conformalize import ConformalExpl
from expl import ExplFactory
from torchvision.models import resnet50, ResNet50_Weights

if __name__ == '__main__':
    args = get_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model = resnet50(weights = ResNet50_Weights.DEFAULT).eval().cuda()

    # TODO: make data loader
    # orig_img_path = "/home/juhyeon/Imagenet/train/n02100236/n02100236_18.JPEG" # bird
    # orig_img_path = "/home/juhyeon/Imagenet/train/n01443537/n01443537_605.JPEG" # dog
    # orig_img_path = "/home/juhyeon/Imagenet/train/n01614925/n01614925_13.JPEG" # fish

    orig_img = Image.open(args.img_path)
    expl_func = ExplFactory().get_explainer(model = model, expl_method = args.expl_method)
    conformalizer = ConformalExpl(orig_img, expl_func, args)

    conformalizer.make_confidence_set()
    conformalizer.evaluate()
    conformalizer.logging()
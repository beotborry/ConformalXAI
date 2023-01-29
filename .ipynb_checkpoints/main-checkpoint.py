#%%
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from transform_factory import center_crop, imagenet_normalize, get_tta_transform
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
from arguments import get_args
from tqdm import tqdm
from conformalize import conformality_score, get_conf_interval, calc_coverage_prob
from sklearn.utils.fixes import sp_version, parse_version 
from sklearn.linear_model import QuantileRegressor
from interpretation import get_grad_cam


if __name__ == '__main__':
    args = get_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model = torchvision.models.resnet50(pretrained=True).eval().cuda()

    # TODO: make data loader
    orig_img_path = "/home/juhyeon/Imagenet/train/n02100236/n02100236_18.JPEG"
    # orig_img_path = "/home/juhyeon/Imagenet/train/n01443537/n01443537_605.JPEG"
    # orig_img = Image.open("/home/juhyeon/Imagenet/train/n01443537/n01443537_605.JPEG")
    orig_img = Image.open(orig_img_path)
    attr_results = []

    for _ in tqdm(range(args.n_sample)):
        tta, inv_tta = get_tta_transform()

        img = tta(imagenet_normalize(center_crop(orig_img)))
        noise = args.sigma * torch.randn_like(img)
        img -= noise

        img = img.unsqueeze(0).cuda()

        upsampled_attr = get_grad_cam(model, img)
        
        upsampled_attr = inv_tta(upsampled_attr)

        upsampled_attr = upsampled_attr.detach().squeeze(0).cpu().numpy()
        

        img = inv_tta(img)
    
        attr_results.append(upsampled_attr)

    attr_results = np.stack(attr_results)


    img = imagenet_normalize(center_crop(orig_img)).unsqueeze(0).cuda()
    true_proxy = get_grad_cam(model, img)
    true_proxy = true_proxy.detach().squeeze(0).cpu().numpy()

    q_hat = conformality_score(attr_results, true_proxy, args.alpha)
    # expl_hat, q_hat = conformality_score(attr_results, args.alpha)



    attr_results = []
    for _ in tqdm(range(args.n_sample)):
        tta, inv_tta = get_tta_transform()

        img = tta(imagenet_normalize(center_crop(orig_img)))
        noise = args.sigma * torch.randn_like(img)
        img -= noise
        img = img.unsqueeze(0).cuda()

        gc = LayerGradCam(model, model.layer4[2].conv3)
        attr = gc.attribute(img, target = model(img).argmax())

        upsampled_attr = LayerAttribution.interpolate(attr, img.shape[2:], 'bilinear')
        
        upsampled_attr = inv_tta(upsampled_attr)

        upsampled_attr = upsampled_attr.detach().squeeze(0).cpu().numpy()
        

        img = inv_tta(img)
    
        attr_results.append(upsampled_attr)

    attr_results = np.stack(attr_results)


    conf_low, conf_high = get_conf_interval(attr_results, q_hat)

    coverage_map= calc_coverage_prob(true_proxy, conf_low, conf_high)

    plt.imshow(coverage_map.squeeze(), cmap='hot', interpolation='nearest')
    plt.text(180, 200, f"mean: {coverage_map.squeeze().mean():.3f}")
    plt.text(180, 180, f"max: {coverage_map.squeeze().max():.3f}")
    plt.text(180, 220, f"min: {coverage_map.squeeze().min():.3f}")
    plt.colorbar()
    plt.savefig(f"./{os.path.basename(orig_img_path)}_alpha_{args.alpha}_n_sample_{args.n_sample}.jpg")


        # _ = viz.visualize_image_attr(np.transpose(upsampled_attr, (1,2,0)),
        #                      np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
        #                      "blended_heat_map",
        #                      sign="all",
        #                      show_colorbar=True)



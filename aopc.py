from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os
from transform_factory import tensorize, center_crop_224, resize_322, imagenet_normalize, resize_224
from PIL import Image
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser

class ConfAOPCTestor():
    def __init__(self, model) -> None:
        self.model = model.cuda()
        self.softmax = torch.nn.Softmax(dim = 1)

    @staticmethod
    def perturbation(expl, img, conf_high, conf_low, mode='insertion'):
        mask = torch.where(torch.logical_and(expl > 0, conf_low > 0), torch.ones_like(expl), 0)
        ratio = mask.flatten(1).sum(1) / (mask.shape[2] * mask.shape[3])
        order = expl.flatten(1).argsort(descending=True)
        
        n_perturb = (ratio * order.shape[1]).type(torch.LongTensor).squeeze()
        n_order = order[range(len(expl)), n_perturb]
        threshold = expl.flatten(1)[range(len(expl)), n_order]
        base_mask = expl > threshold.reshape(len(expl), 1, 1).unsqueeze(1)

        return (base_mask * img).detach(), (mask * img).detach()

    def test_step(self, expl, img, label, conf_high, conf_low, mode='insertion'):

        img_base, img_our = self.perturbation(expl, img, conf_high, conf_low, mode=mode)

        plt.imshow(img_base[3].sum(0))
        plt.show()
        plt.imshow(img_our[3].sum(0))

        logit = self.model(img_base.cuda())
        del img_base
        prob_base = self.softmax(logit)

        aopc_prob_base = prob_base[range(len(label)), label].detach().mean()

        logit = self.model(img_our.cuda())
        del img_our
        prob_our = self.softmax(logit)
        aopc_prob_our = prob_our[range(len(label)), label].detach().mean()

        print(aopc_prob_base, aopc_prob_our)

        # aopc_prob = prob[range(len(label)), label].detach().mean()
        # prob_list.append(aopc_prob.detach().cpu())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_data", type=int)
    parser.add_argument("--expl_method")
    parser.add_argument("--dataset", default="center_crop_224")
    parser.add_argument("--orig_input_method", default="center_crop_224")

    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed
    num_data = args.num_data
    expl_method = args.expl_method
    orig_input_method = args.orig_input_method

    batch_size = 128
    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

    tester = ConfAOPCTestor(model)
    with open(f"./val_{dataset}_seed_{seed}.npy", "rb") as f:
        filepath_list = np.load(f)

    for i in range(num_data // batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        orig_imgs = []
        orig_expls = []
        for img_path in filepath_list[start:end]:
            img_name = os.path.basename(img_path)

            orig_img_pil = Image.open(img_path)
            orig_img = imagenet_normalize(tensorize(center_crop_224(resize_322(orig_img_pil))))

            orig_imgs.append(orig_img)

            with open(f"results/val_seed_{seed}_dataset_{dataset}_orig_input_method_{orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_orig_true_config.npy", "rb") as f:
                orig_expl = np.load(f, allow_pickle=True)
                true_expls = np.load(f, allow_pickle=True)
                configs = np.load(f, allow_pickle=True)

                orig_expl = F.interpolate(torch.tensor(orig_expl).unsqueeze(0), (224, 224), mode='bicubic').squeeze(0).numpy()
                orig_expls.append(orig_expl)

        orig_imgs = torch.stack(orig_imgs)
        orig_expls = torch.tensor(np.stack(orig_expls))

        y = model(orig_imgs.cuda()).argmax(dim = 1)
        conf_highs = []
        conf_lows = []

        alpha = 0.05

        for img_path in filepath_list[start:end]:
            img_name = os.path.basename(img_path)
            
            with open(f"results/val_seed_{seed}_dataset_{dataset}_orig_input_method_{orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_results.pkl", "rb") as f:
                results = np.load(f, allow_pickle=True)

            result = results[0]

            conf_highs.append(torch.tensor(result['conf_high']))
            conf_lows.append(torch.tensor(result['conf_low']))


        conf_highs = torch.stack(conf_highs)
        conf_lows = torch.stack(conf_lows)
        
        tester.test_step(orig_expls, orig_imgs, y, conf_highs, conf_lows)
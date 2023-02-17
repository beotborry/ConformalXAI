from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os
from transform_factory import tensorize, center_crop_224, resize_322, imagenet_normalize, resize_224
from PIL import Image
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from transform_factory import get_spatial_transform, get_color_transform, ToPIL, PIL2Tensor, gauss_noise_tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class ConfAOPCTestor():
    def __init__(self, model) -> None:
        self.model = model.cuda()
        self.softmax = torch.nn.Softmax(dim = 1)

    @staticmethod
    def perturbation(expl, img, r,  conf_high, conf_low, mode='insertion'):
        mask = torch.where(torch.logical_and(expl > 0, conf_low > 0), torch.ones_like(expl), 0)
        ratio = mask.flatten(1).sum(1) / (mask.shape[2] * mask.shape[3])
    
        # base mask generating
        order = expl.flatten(1).argsort(descending=True)
        n_perturb = (r * ratio * order.shape[1]).type(torch.LongTensor).squeeze()
        n_order = order[range(len(expl)), n_perturb]
        threshold = expl.flatten(1)[range(len(expl)), n_order]
        base_mask = expl > threshold.reshape(len(expl), 1, 1).unsqueeze(1)

        # our mask generating
        order = (mask * expl).flatten(1).argsort(descending=True)
        # order = (mask * conf_high).flatten(1).argsort(descending=True)
        n_perturb = (r * ratio * order.shape[1]).type(torch.LongTensor).squeeze()
        n_order = order[range(len(expl)), n_perturb]
        threshold = (mask * expl).flatten(1)[range(len(expl)), n_order]
        # threshold = (mask * conf_high).flatten(1)[range(len(expl)), n_order]

        our_mask = (mask * expl) > threshold.reshape(len(expl), 1, 1).unsqueeze(1)
        # our_mask = (mask * conf_high) > threshold.reshape(len(expl), 1, 1).unsqueeze(1)


        return (base_mask * img).detach(), (our_mask * img).detach()

    def test_step(self, expl, img, label, conf_high, conf_low, mode='insertion', transform=None, config=None):

        base_prob_list = []
        our_prob_list = []
        for r in np.arange(0, 1.05, 0.05):
            img_base, img_our = self.perturbation(expl, img, r, conf_high, conf_low, mode=mode)

            if transform is not None:
                for i in range(len(config)):
                    if "spatial" in transform:
                        t = transforms.Compose([
                            transforms.RandomHorizontalFlip(configs[i]['flip_horizon']),
                            transforms.RandomVerticalFlip(configs[i]['flip_vertical']),
                            transforms.RandomRotation((configs[i]['rot_angle'], configs[i]['rot_angle']), InterpolationMode.BILINEAR),
                        ])
                        img_base[i] = t(img_base[i])
                        img_our[i] = t(img_our[i])

            logit = self.model(img_base.cuda())
            del img_base
            prob_base = self.softmax(logit)
            del logit

            base_prob_list.append(prob_base[range(len(label)), label].detach().mean().cpu())

            del prob_base

            logit = self.model(img_our.cuda())
            del img_our
            prob_our = self.softmax(logit)
            our_prob_list.append(prob_our[range(len(label)), label].detach().mean().cpu())
            del prob_our

            print(r, base_prob_list[-1], our_prob_list[-1])
        return base_prob_list, our_prob_list

class AOPCTestor():
    def __init__(self, model) -> None:
        self.model = model.cuda()
        self.softmax = torch.nn.Softmax(dim = 1)


    @staticmethod
    def perturbation(expl, img, ratio, mode="insertion"):
    # expl : [B, C=1, H, W]
    # img : [B, C=3, H, W]
        if mode == "insertion":
            order = expl.flatten(1).argsort(descending=True)
            n_perturb = int(ratio * order.shape[1])
            n_order = order[:, n_perturb] 
            threshold = expl.flatten(1)[range(len(expl)), n_order]
            mask = expl > threshold.reshape(len(expl), 1, 1).unsqueeze(1)
        elif mode == "deletion":
            order = expl.flatten(1).argsort()
            n_perturb = int(ratio * order.shape[1])
            n_order = order[:, n_perturb]
            threshold = expl.flatten(1)[range(len(expl)), n_order]
            mask = expl > threshold.reshape(len(expl), 1, 1).unsqueeze(1)        
            
        return (img * mask).detach()

    def test_step(self, expl, img, label, mode="insertion", transform=None, configs=None):
        prob_list = []

        for ratio in np.arange(0, 1, 0.05):

            img_p = self.perturbation(expl, img, ratio=ratio, mode=mode)
            if transform is not None:
                for i, _img in enumerate(img_p):
                    if "spatial" in transform:
                        t = transforms.Compose([
                            transforms.RandomHorizontalFlip(configs[i]['flip_horizon']),
                            transforms.RandomVerticalFlip(configs[i]['flip_vertical']),
                            transforms.RandomRotation((configs[i]['rot_angle'], configs[i]['rot_angle']), InterpolationMode.BILINEAR),
                        ])
                        img_p[i] = t(_img)

            logit = self.model(img_p.cuda())
            del img_p
            prob = self.softmax(logit)

            aopc_prob = prob[range(len(label)), label].detach().mean()
            prob_list.append(aopc_prob.detach().cpu())

        return prob_list


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_data", type=int)
    parser.add_argument("--expl_method")
    parser.add_argument("--dataset", default="center_crop_224")
    parser.add_argument("--orig_input_method", default="center_crop_224")
    parser.add_argument("--mode", choices=['insertion', 'deletion'])
    parser.add_argument("--tester", choices=['OrigAOPC', 'ConfAOPC', 'ConfAOPC_high'])
    parser.add_argument("--transform", type=str, nargs="+", default=None)

    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed
    num_data = args.num_data
    expl_method = args.expl_method
    orig_input_method = args.orig_input_method

    batch_size = 100
    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

    log_name_base = f"./aopc_results/{args.tester}_transform_{args.transform}_mode_{args.mode}_expl_method_{expl_method}_seed_{args.seed}"
    print(vars(args))

    if args.tester == 'ConfAOPC' or args.tester == "ConfAOPC_high":
        tester = ConfAOPCTestor(model)
    elif args.tester == 'OrigAOPC':
        tester = AOPCTestor(model)

    with open(f"./val_{dataset}_seed_{seed}.npy", "rb") as f:
        filepath_list = np.load(f)

    for i in range(num_data // batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        orig_imgs = []
        orig_expls = []
        T_spatial_configs = []
        for img_path in filepath_list[start:end]:


            img_name = os.path.basename(img_path)
            
            if os.path.exists(f"results/recent_results/val_seed_{seed}_dataset_{dataset}_orig_input_method_{orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_orig_true_config.npy") == False:
                continue

            orig_img_pil = Image.open(img_path)


            if args.transform is not None:
            
                orig_img = imagenet_normalize(tensorize(center_crop_224(resize_322(orig_img_pil))))
                y = model(orig_img.unsqueeze(0).cuda()).argmax(dim = 1)

                while True:                
                    with torch.no_grad():
                        T_spatial, T_inv_spatial, config = get_spatial_transform()
                        T_color = get_color_transform()

                        if "noise" in args.transform:
                            _transformed_img = ToPIL(gauss_noise_tensor(PIL2Tensor(center_crop_224(resize_322(orig_img_pil))))) 
                        else:
                            _transformed_img = center_crop_224(resize_322(orig_img_pil))

                        if "color" in args.transform and "spatial" in args.transform:
                            transformed_img = imagenet_normalize(tensorize(T_spatial(T_color(_transformed_img))))
                        elif "color" in args.transform:
                            transformed_img = imagenet_normalize(tensorize(T_color(_transformed_img)))
                        elif "spatial" in args.transform:
                            transformed_img = imagenet_normalize(tensorize(T_spatial(_transformed_img)))

                        logit = model(transformed_img.unsqueeze(0).cuda())

                        if y == logit.argmax(dim = 1):                    
                            if "color" in args.transform:
                                omit_spatial = imagenet_normalize(tensorize(T_color(_transformed_img)))
                            else:
                                omit_spatial = imagenet_normalize(tensorize(_transformed_img))

                            orig_imgs.append(omit_spatial)
                        
                            if "spatial" in args.transform:
                                T_spatial_configs.append(config)
                            break

            else:
                orig_img = imagenet_normalize(tensorize(center_crop_224(resize_322(orig_img_pil))))
                orig_imgs.append(orig_img)

            
            with open(f"results/recent_results/val_seed_{seed}_dataset_{dataset}_orig_input_method_{orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_orig_true_config.npy", "rb") as f:
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
            

            try:
                with open(f"results/recent_results/val_seed_{seed}_dataset_{dataset}_orig_input_method_{orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_results.pkl", "rb") as f:
                    results = np.load(f, allow_pickle=True)

                result = results[0]

                conf_highs.append(torch.tensor(result['conf_high']))
                conf_lows.append(torch.tensor(result['conf_low']))
            except:
                continue


        conf_highs = torch.stack(conf_highs)
        conf_lows = torch.stack(conf_lows)
        
        # tester.test_step(orig_expls, orig_imgs, y, conf_highs, conf_lows)

        if args.tester == 'OrigAOPC':
            print(orig_expls.shape, orig_imgs.shape)
            orig_prob_list = torch.stack(tester.test_step(orig_expls, orig_imgs, y, args.mode, args.transform, T_spatial_configs))
            high_ins_list = torch.stack(tester.test_step(conf_highs, orig_imgs, y, args.mode, args.transform, T_spatial_configs))
            low_ins_list = torch.stack(tester.test_step(conf_lows, orig_imgs, y, args.mode, args.transform, T_spatial_configs))

            log_name = log_name_base + f"_batch_num_{i}.pt"
            torch.save(torch.vstack((orig_prob_list, high_ins_list, low_ins_list)), log_name)

        elif args.tester == "ConfAOPC" or args.tester == "ConfAOPC_high":
            orig_prob_list, our_prob_list = tester.test_step(orig_expls, orig_imgs, y, conf_highs, conf_lows, transform=args.transform, config=T_spatial_configs)
            orig_prob_list = torch.stack(orig_prob_list)
            our_prob_list = torch.stack(our_prob_list)

            log_name = log_name_base + f"_batch_num_{i}.pt"
            torch.save(torch.vstack((orig_prob_list, our_prob_list)), log_name)
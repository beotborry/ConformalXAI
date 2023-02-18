from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os
from transform_factory import tensorize, center_crop_224, resize_322, imagenet_normalize, resize_224
from PIL import Image
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from transform_factory import get_spatial_transform, get_color_transform, ToPIL, PIL2Tensor, gauss_noise_tensor, get_trivial_augment
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def load_results(filename, alpha):
    with open(filename, "rb") as f:
        results = np.load(f, allow_pickle=True)

    idx = int(alpha // 0.05) - 1
    
    result = results[idx]

    return result


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
        n_perturb = (r * ratio * order.shape[1]).type(torch.LongTensor).squeeze()
        n_order = order[range(len(expl)), n_perturb]
        threshold = (mask * expl).flatten(1)[range(len(expl)), n_order]
        our_mask = (mask * expl) > threshold.reshape(len(expl), 1, 1).unsqueeze(1)


        return (base_mask * img).detach(), (our_mask * img).detach()

    def test_step(self, expl, img, label, conf_high, conf_low, mode='insertion', transform=None, configs=None):
        base_prob_list = []
        our_prob_list = []
        for r in np.arange(0, 1, 0.05):
            img_base, img_our = self.perturbation(expl, img, r, conf_high, conf_low, mode=mode)
            if transform is not None:
                if "spatial" in transform:
                    for idx, config in enumerate(configs):
                        t = transforms.Compose([
                            transforms.RandomHorizontalFlip(config['flip_horizon']),
                            transforms.RandomRotation((config['rot_angle'], config['rot_angle']), InterpolationMode.BILINEAR),
                        ])
                        img_base[idx] = t(img_base[idx])
                        img_our[idx] = t(img_our[idx])

            logit = self.model(img_base.cuda())
            del img_base
            prob_base = self.softmax(logit)
            del logit

            base_prob_list.append(prob_base[:, label[0]].detach().sum().cpu())

            del prob_base

            logit = self.model(img_our.cuda())
            del img_our
            prob_our = self.softmax(logit)
            our_prob_list.append(prob_our[:, label[0]].detach().sum().cpu())

            del prob_our
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

        print(configs)
        for ratio in np.arange(0, 1, 0.05):

            img_p = self.perturbation(expl, img, ratio=ratio, mode=mode)
            if transform is not None:
                for i, _img in enumerate(img_p):
                    if "spatial" in transform:
                        t = transforms.Compose([
                            transforms.RandomHorizontalFlip(configs['flip_horizon']),
                            transforms.RandomRotation((configs['rot_angle'], configs['rot_angle']), InterpolationMode.BILINEAR),
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

        img_names = []
        conf_highs = []
        conf_lows = []
        cal_indices = []
        val_indices = []
        transform_configs =[]

        alpha = 0.1

        # load results
        for img_path in filepath_list[start:end]:
            img_name = os.path.basename(img_path)
            try:
                result = load_results(f"results/val_seed_{seed}_dataset_{dataset}_orig_input_method_{orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_results.pkl", alpha)
                img_names.append(result['img'])
                conf_highs.append(torch.tensor(result['conf_high']))
                conf_lows.append(torch.tensor(result['conf_low']))
                cal_indices.append(torch.tensor(result['cal_idx']))
                val_indices.append(torch.tensor(result['val_idx']))
            except:
                continue

        conf_highs = torch.stack(conf_highs)
        conf_lows = torch.stack(conf_lows)
        cal_indices = torch.stack(cal_indices)
        val_indices = torch.stack(val_indices)

        print(conf_highs.shape, conf_lows.shape, cal_indices.shape, val_indices.shape)

        for img_path in filepath_list[start:end]:
            img_name = os.path.basename(img_path)
            idx = 0
            
            if not os.path.exists(f"results/val_seed_{seed}_dataset_{dataset}_orig_input_method_{orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_orig_true_config.npy"):
                continue

            orig_img_pil = Image.open(img_path)
            orig_img = center_crop_224(resize_322(orig_img_pil))
            orig_imgs.append(orig_img)
            
            with open(f"results/val_seed_{seed}_dataset_{dataset}_orig_input_method_{orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_orig_true_config.npy", "rb") as f:
                orig_expl = np.load(f, allow_pickle=True)
                true_expls = np.load(f, allow_pickle=True)

                orig_expl = F.interpolate(torch.tensor(orig_expl).unsqueeze(0), (224, 224), mode='bicubic').squeeze(0).numpy()
                orig_expls.append(orig_expl)

            with open(f"results/val_seed_{seed}_dataset_{dataset}_orig_input_method_{orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_transform_config.txt", "r") as f:
                config = np.array(f.readlines())[val_indices[idx]]
                transform_configs.append(config)
            idx += 1

        orig_expls = torch.tensor(np.stack(orig_expls))
        transform_configs = np.stack(transform_configs)
        print(transform_configs.shape)

 

        if args.tester == 'OrigAOPC':
            print(orig_expls.shape, conf_highs.shape, conf_lows.shape, orig_imgs.shape)
            orig_prob_list = torch.stack(tester.test_step(orig_expls, orig_imgs, y, args.mode, args.transform, T_spatial_configs))


            high_ins_list = torch.stack(tester.test_step(conf_highs, orig_imgs, y, args.mode, args.transform, T_spatial_configs))
            low_ins_list = torch.stack(tester.test_step(conf_lows, orig_imgs, y, args.mode, args.transform, T_spatial_configs))


            log_name = log_name_base + f"_batch_num_{i}.pt"
            torch.save(torch.vstack((orig_prob_list, high_ins_list, low_ins_list)), log_name)

        elif args.tester == "ConfAOPC":
            for img_name, orig_expl, orig_img, conf_high, conf_low, configs in zip(img_names, orig_expls, orig_imgs, conf_highs, conf_lows, transform_configs):
                _orig_expl = orig_expl.unsqueeze(0)
                _conf_high = conf_high.unsqueeze(0)
                _conf_low = conf_low.unsqueeze(0)

                _orig_probs = torch.zeros(20)
                _our_probs = torch.zeros(20)
                
                imgs = []
                spatial_configs = []
                
                y = model(imagenet_normalize(tensorize(orig_img)).unsqueeze(0).cuda()).argmax(dim = 1).unsqueeze(0)

                for i, config in enumerate(tqdm(configs)):
                    config = eval(config)
                    T_color = get_trivial_augment(config = config, color_only=True)
                    _orig_img = T_color(orig_img)
                    imgs.append(_orig_img)

                    config = dict(config)
                    T_spatial_config = {
                        'flip_horizon': config['hflip'],
                    }
                    if 'Rotate' in config.keys():
                        T_spatial_config.update({
                            'rot_angle': config['Rotate']
                        })
                    else:
                        T_spatial_config.update({
                            'rot_angle': 0
                        })

                    spatial_configs.append(T_spatial_config)
                    if (i + 1) % 128 == 0:
                        imgs = torch.stack(imgs)
                        orig_prob_list, our_prob_list = tester.test_step(_orig_expl.repeat(128, 1, 1, 1), imgs, y, _conf_high.repeat(128, 1, 1, 1), _conf_low.repeat(128, 1, 1, 1), transform=['spatial'], configs=spatial_configs)

                        orig_prob_list = torch.stack(orig_prob_list)
                        our_prob_list = torch.stack(our_prob_list)

                        _orig_probs += orig_prob_list
                        _our_probs += our_prob_list

                        imgs = []
                        spatial_configs = []

                        print(_orig_probs.shape, _our_probs.shape)
                        break
                    
                _orig_probs /= 128
                _our_probs /= 128
                print(_orig_probs, _our_probs)

                log_name = log_name_base + f"_{img_name}.pt"
                torch.save(torch.vstack((_orig_probs, _our_probs)), log_name)
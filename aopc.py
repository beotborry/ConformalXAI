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
from utils import set_seed
from torch.nn import DataParallel



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
    def perturbation(expl, img, r, conf_high, conf_low, cal_avg, mode='insertion'):
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

        # cal avg mask
        order = cal_avg.flatten(1).argsort(descending=True)
        n_perturb = (r * ratio * order.shape[1]).type(torch.LongTensor).squeeze()
        n_order = order[range(len(expl)), n_perturb]
        threshold = cal_avg.flatten(1)[range(len(expl)), n_order]
        cal_avg_mask = cal_avg > threshold.reshape(len(expl), 1, 1).unsqueeze(1)

        return (base_mask * img).detach(), (our_mask * img).detach(), (cal_avg_mask * img).detach()

    def test_step(self, expl, img, off_center_imgs, label, conf_high, conf_low, cal_avg, mode='insertion', transform=None, configs=None):
        base_prob_list = []
        our_prob_list = []
        avg_prob_list = []
        for r in np.arange(0, 1.05, 0.1):
            img_base, img_our, img_avg = self.perturbation(expl, img, r, conf_high, conf_low, cal_avg,  mode=mode)
            if transform is not None:
                if "spatial" in transform:
                    for idx, config in enumerate(configs):
                        t = transforms.Compose([
                            transforms.RandomHorizontalFlip(config['flip_horizon']),
                            transforms.RandomRotation((config['rot_angle'], config['rot_angle']), InterpolationMode.BILINEAR),
                        ])
                        img_322 = off_center_imgs[idx]
                        img_322[:, 49:273, 49:273] = img_base[idx]
                        img_base[idx] = center_crop_224(t(img_322))

                        img_322[:, 49:273, 49:273] = img_our[idx]
                        img_our[idx] = center_crop_224(t(img_322))

                        img_322[:, 49:273, 49:273] = img_avg[idx]
                        img_avg[idx] = center_crop_224(t(img_322))


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

            logit = self.model(img_avg.cuda())
            del img_avg
            prob_avg = self.softmax(logit)
            avg_prob_list.append(prob_avg[:, label[0]].detach().sum().cpu())

            print(base_prob_list[-1], our_prob_list[-1], avg_prob_list[-1])
        return base_prob_list, our_prob_list, avg_prob_list

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
    parser.add_argument("--perturb_num", type=int)
    parser.add_argument("--perturb_iter", type=int, default= 4)
    parser.add_argument("--device", type=int)



    args = parser.parse_args()

    set_seed(777)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    dataset = args.dataset
    seed = args.seed
    num_data = args.num_data
    expl_method = args.expl_method
    orig_input_method = args.orig_input_method
    region = (49, 49, 273, 273)

    batch_size = 100
    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    model = DataParallel(model, device_ids=[args.device, args.device + 1])

    log_name_base = f"./aopc_results/{args.tester}_transform_{args.transform}_mode_{args.mode}_expl_method_{expl_method}_seed_{args.seed}_perturb_num_{args.perturb_num * args.perturb_iter}"
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
        cal_averages = []

        alpha = 0.05

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
                cal_averages.append(torch.tensor(result['cal_average']))
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
            # orig_img = center_crop_224(resize_322(orig_img_pil))
            orig_img = resize_322(orig_img_pil)
            orig_imgs.append(orig_img)
            
            with open(f"results/val_seed_{seed}_dataset_{dataset}_orig_input_method_{orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_orig_true_config.npy", "rb") as f:
                orig_expl = np.load(f, allow_pickle=True)
                true_expls = np.load(f, allow_pickle=True)

                orig_expl = F.interpolate(torch.tensor(orig_expl).unsqueeze(0), (224, 224), mode='bicubic').squeeze(0).numpy()
                orig_expls.append(orig_expl)

            idx += 1

        orig_expls = torch.tensor(np.stack(orig_expls))
 

        if args.tester == 'OrigAOPC':
            print(orig_expls.shape, conf_highs.shape, conf_lows.shape, orig_imgs.shape)
            orig_prob_list = torch.stack(tester.test_step(orig_expls, orig_imgs, y, args.mode, args.transform, T_spatial_configs))


            high_ins_list = torch.stack(tester.test_step(conf_highs, orig_imgs, y, args.mode, args.transform, T_spatial_configs))
            low_ins_list = torch.stack(tester.test_step(conf_lows, orig_imgs, y, args.mode, args.transform, T_spatial_configs))


            log_name = log_name_base + f"_batch_num_{i}.pt"
            torch.save(torch.vstack((orig_prob_list, high_ins_list, low_ins_list)), log_name)

        elif args.tester == "ConfAOPC":
            for img_name, orig_expl, orig_img, conf_high, conf_low, cal_average in zip(img_names, orig_expls, orig_imgs, conf_highs, conf_lows, cal_averages):
                
                log_name = log_name_base + f"_{img_name}.pt"
                if os.path.exists(log_name):
                    print("skipped!")
                    continue
                    

                _orig_expl = orig_expl.unsqueeze(0)
                _conf_high = conf_high.unsqueeze(0)
                _conf_low = conf_low.unsqueeze(0)

                _orig_probs = torch.zeros(11)
                _our_probs = torch.zeros(11)
                _avg_probs = torch.zeros(11)
                
                imgs = []
                off_center_imgs = []
                spatial_configs = []
                
                y = model(imagenet_normalize(tensorize(orig_img)).unsqueeze(0).cuda()).argmax(dim = 1).unsqueeze(0)

                total_num = args.perturb_num * args.perturb_iter

                for _ in range(args.perturb_iter):
                    perturbed_num = 0
                    while perturbed_num < args.perturb_num:
                        T_color = get_trivial_augment(aopc = True, trans_opt='color')
                        T_spatial, _, spatial_config = get_spatial_transform()
                        _orig_img = T_color(orig_img)
                        logit = model(center_crop_224(T_spatial(_orig_img)).unsqueeze(0).cuda())
                        if logit.argmax() == y:
                            imgs.append(center_crop_224(_orig_img))
                            off_center_img = _orig_img.clone()
                            off_center_img[:, region[1]:region[3], region[0]:region[2]] = 0
                            off_center_imgs.append(off_center_img)

                            spatial_configs.append(spatial_config)
                            perturbed_num += 1
                        else:
                            continue

                    imgs = torch.stack(imgs)

                    orig_prob_list, our_prob_list, avg_prob_list = tester.test_step(_orig_expl.repeat(args.perturb_num, 1, 1, 1), imgs, off_center_imgs,  y, _conf_high.repeat(args.perturb_num, 1, 1, 1), _conf_low.repeat(args.perturb_num, 1, 1, 1), cal_avg=cal_average.repeat(args.perturb_num, 1, 1, 1), transform=['spatial'], configs=spatial_configs)
                    
                    orig_prob_list = torch.stack(orig_prob_list)
                    our_prob_list = torch.stack(our_prob_list)
                    avg_prob_list = torch.stack(avg_prob_list)

                    _orig_probs += orig_prob_list
                    _our_probs += our_prob_list
                    _avg_probs += avg_prob_list

                    imgs = []
                    spatial_configs = []
                    perturbed_num = 0

                print(_orig_probs / total_num, _our_probs / total_num, _avg_probs / total_num)
                torch.save(torch.vstack((_orig_probs / total_num, _our_probs / total_num, _avg_probs / total_num)), log_name)
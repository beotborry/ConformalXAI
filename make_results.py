import numpy as np
import torch.nn.functional as F
from transform_factory import center_crop_224
from torchvision import transforms
import torch
from tqdm import tqdm
from captum.attr import visualization as viz
import os
from PIL import Image
from transform_factory import tensorize, resize_322, center_crop_224, resize_224
from utils import set_seed
from argparse import ArgumentParser
import pickle
from torchvision.transforms import InterpolationMode

def calc_score_and_test_expls(true_expls, orig_expl, configs):
    indicies = np.arange(0, 2000, 1)
    np.random.shuffle(indicies)

    cal_idx, val_idx = indicies[:1000], indicies[1000:]

    scores = []

    for true_expl, config in zip(true_expls[cal_idx], configs[cal_idx]):

        T_inv_spatial = transforms.Compose([
            transforms.RandomRotation((-config['rot_angle'], -config['rot_angle']), InterpolationMode.BILINEAR),
            transforms.RandomVerticalFlip(config['flip_vertical']),
            transforms.RandomHorizontalFlip(config['flip_horizon']),
            
        ])
        
        # true_expl = center_crop_224(resize_322(T_inv_spatial(resize_224(torch.tensor(true_expl).cuda().unsqueeze(0))))).squeeze(0)

        # true_expl = center_crop_224(F.interpolate(T_inv_spatial(torch.tensor(true_expl).cuda().unsqueeze(0)), (322, 322), mode='bicubic')).squeeze(0)
        true_expl = center_crop_224(T_inv_spatial(F.interpolate(torch.tensor(true_expl).cuda().unsqueeze(0), (322, 322), mode='bicubic'))).squeeze(0)
        scores.append(torch.abs(true_expl - orig_expl))
    scores = torch.stack(scores)


    test_expls =[]
    for true_expl, config in zip(true_expls[val_idx], configs[val_idx]):

        T_inv_spatial = transforms.Compose([
            transforms.RandomRotation((-config['rot_angle'], -config['rot_angle']), InterpolationMode.BILINEAR),
            transforms.RandomVerticalFlip(config['flip_vertical']),
            transforms.RandomHorizontalFlip(config['flip_horizon']),
            
        ])
        # test_expls.append(center_crop_224(resize_322(T_inv_spatial(resize_224(torch.tensor(true_expl).cuda().unsqueeze(0))))).squeeze(0))
        # test_expls.append(center_crop_224(F.interpolate(T_inv_spatial(torch.tensor(true_expl).cuda().unsqueeze(0)), (322, 322), mode='bicubic')).squeeze(0))
        test_expls.append(center_crop_224(T_inv_spatial(F.interpolate(torch.tensor(true_expl).cuda().unsqueeze(0), (322, 322), mode='bicubic'))).squeeze(0))

    test_expls = torch.stack(test_expls)

    return scores, test_expls

def qhat(score, alpha:float):
    n = score.shape[0]
    q_hat = torch.quantile(score, np.ceil((n+1) * (1-alpha)) / n, axis = 0)

    return q_hat

def get_conf_interval(expl, q_hat):
    high = expl + q_hat
    low = expl - q_hat
    return (low, high)

def calc_coverage_prob(true, conf_low, conf_high):
    is_cover = torch.logical_and(conf_low <= true, true <= conf_high)
    coverage_prob = torch.sum(is_cover, axis = 0) / true.shape[0]

    return coverage_prob

def zero_contain_rate(conf_high, conf_low):
    zeros = torch.zeros_like(conf_high)
    contain_zero = torch.where(torch.logical_and(zeros > conf_low, zeros < conf_high))

    return len(contain_zero[0]) / (zeros.shape[1] * zeros.shape[2])
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expl_method", choices=["GradCAM", "LayerIG", "LayerXAct", "LayerDL"])
    parser.add_argument("--dataset", choices=["center_crop_224", "resize_224"])
    parser.add_argument("--orig_input_method", choices=["center_crop_224", "resize_224"])

    args = parser.parse_args()
    seed = args.seed
    expl_method = args.expl_method
with open(f"./val_{args.dataset}_seed_{seed}.npy", "rb") as f:
    filepath_list = np.load(f)

for img_path in tqdm(filepath_list):
    img_name = os.path.basename(img_path)


    expr_path = f"results/val_seed_{seed}_dataset_{args.dataset}_orig_input_method_{args.orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_orig_true_config.npy" 
    results_path = f"results/val_seed_{seed}_dataset_{args.dataset}_orig_input_method_{args.orig_input_method}_pred_orig_eval_orig_transform_both_sign_all_reduction_sum/{img_name}_expl_{expl_method}_sample_2000_sigma_0.05_seed_{seed}_results.pkl"

    if os.path.exists(results_path):
        continue

    try:
        with open(expr_path, "rb") as f:
            orig_expl = np.load(f, allow_pickle=True)
            true_expls = np.load(f, allow_pickle=True)
            configs = np.load(f, allow_pickle=True)
    except:
        continue

    if args.orig_input_method == "center_crop_224":
        orig_expl = F.interpolate(torch.tensor(orig_expl).cuda().unsqueeze(0), (224, 224), mode='bicubic').squeeze()
    else:
        orig_expl = center_crop_224(F.interpolate(torch.tensor(orig_expl).cuda().unsqueeze(0), (322, 322), mode='bicubic')).squeeze()

    scores, test_expls = calc_score_and_test_expls(true_expls, orig_expl, configs)


    results = []

    for alpha in np.arange(0.05, 1, 0.05):
        q_hat = qhat(scores, alpha)

        conf_low, conf_high = get_conf_interval(orig_expl, q_hat)
    
        
        coverage_prob = calc_coverage_prob(test_expls, conf_low, conf_high)

        zc_rate = zero_contain_rate(conf_high, conf_low)
        
        results.append({
            'img': img_name,
            'expl_method': expl_method,
            'alpha': alpha,
            'coverage_prob': coverage_prob.detach().cpu(),
            'zero_contain_rate': zc_rate,
            'orig_expl': orig_expl.detach().cpu(),
            'conf_high': conf_high.detach().cpu(),
            'conf_low': conf_low.detach().cpu()
        })

    with open(results_path, "wb") as f:
        pickle.dump(results, f)


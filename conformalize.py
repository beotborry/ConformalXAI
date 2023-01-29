import torch
import numpy as np
import matplotlib.pyplot as plt
from transform_factory import resize_322, center_crop_224, tensorize, get_spatial_transform, get_color_transform, imagenet_normalize
from logger import Logger
from tqdm import tqdm

class ConformalExpl:
    def __init__(self, orig_img, expl_func, args) -> None:
        self.orig_img = orig_img
        self.expl_func = expl_func
        self.n_sample = args.n_sample
        self.alpha = args.alpha
        self.noise_sigma = args.sigma
        self.logger = Logger(args)
        self.orig_expl = center_crop_224(expl_func(tensorize(resize_322(orig_img)).unsqueeze(0).cuda()))

    def make_confidence_set(self):
        print("Make confidence set ...")
        true_expls = []
        pred_expls = []
        for _ in tqdm(range(self.n_sample)):
            T_spatial, T_inv_spatial = get_spatial_transform()
            T_color = get_color_transform()

            tensorized_img = tensorize(resize_322(self.orig_img))

            noise = self.noise_sigma * torch.randn_like(tensorized_img)

            transformed_img = T_spatial(imagenet_normalize(T_color(tensorized_img - noise)))
            transformed_img = transformed_img.unsqueeze(0).cuda()

            true_expl = center_crop_224(T_inv_spatial(self.expl_func(transformed_img)))
            true_expl = true_expl.detach().squeeze(0).cpu().numpy()
            
            # Use \phi(X^*) as predictor
            pred_expl = center_crop_224(self.expl_func(imagenet_normalize(T_color(tensorized_img)).unsqueeze(0).cuda())) 
            
            # Use \phi(X_obs) as predictor
            # pred_expl = center_crop_224(expl_func(imagenet_normalize(tensorized_img).unsqueeze(0).cuda()))
        
            pred_expl = pred_expl.detach().squeeze(0).cpu().numpy()

            true_expls.append(true_expl)
            pred_expls.append(pred_expl)
            
        true_expls = np.stack(true_expls)
        pred_expls = np.stack(pred_expls)

        self.q_hat = self.conformality_score(true_expls, pred_expls, self.alpha)


    def evaluate(self):
        print("Evaluating ...")
        test_expls = []

        for _ in tqdm(range(self.n_sample)):
            T_spatial, T_inv_spatial = get_spatial_transform()
            T_color = get_color_transform()

            tensorized_img = tensorize(resize_322(self.orig_img))
            noise = self.noise_sigma * torch.randn_like(tensorized_img)

            transformed_img = T_spatial(imagenet_normalize(T_color(tensorized_img - noise)))
            transformed_img = transformed_img.unsqueeze(0).cuda()

            transformed_expl = center_crop_224(T_inv_spatial(self.expl_func(transformed_img)))
            transformed_expl = transformed_expl.detach().squeeze(0).cpu().numpy()

            test_expls.append(transformed_expl)

        test_expls = np.stack(test_expls)

        self.conf_low, self.conf_high = self.get_conf_interval(self.orig_expl.detach().squeeze(0).cpu().numpy(), self.q_hat)
        self.coverage_map = self.calc_coverage_prob(test_expls, self.conf_low, self.conf_high)

    def logging(self):
        self.logger.save_conf_interval(self.orig_expl.detach().squeeze(0).cpu().numpy(), self.conf_low, self.conf_high)
        self.logger.save_histogram(self.coverage_map)
        self.logger.save_coverage_map(self.coverage_map, self.conf_low, self.conf_high)


    def conformality_score(self, true: np.ndarray, pred: np.ndarray, alpha: float):
        '''
        expls: [n_sample, 1, H, W] for Grad-CAM 
        '''
        assert true.shape == pred.shape

        n = pred.shape[0]

        score = np.abs(true - pred)
        q_hat = np.quantile(score, np.ceil((n+1) * (1-alpha)) / n, axis = 0)

        return q_hat

    def get_conf_interval(self, expl: np.ndarray, q_hat: np.ndarray):
        high = expl + q_hat
        low = expl - q_hat
        return (low, high)

    def calc_coverage_prob(self, true: np.ndarray, conf_low: np.ndarray, conf_high: np.ndarray):
        is_cover = np.logical_and(conf_low <= true, true <= conf_high)
        coverage_prob = np.sum(is_cover, axis = 0) / true.shape[0]

        return coverage_prob
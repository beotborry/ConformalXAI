import torch
import numpy as np
import matplotlib.pyplot as plt
from transform_factory import resize_322, center_crop_224, tensorize, get_spatial_transform, get_color_transform, imagenet_normalize, resize_224, center_crop_32, resize_46, resize_232, TransformFactory
from logger import Logger
from tqdm import tqdm
import time
from transform_factory import get_trivial_augment
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, to_tensor
from utils import entropy

class ConformalExpl:
    def __init__(self, orig_img, expl_func, args, img_path = None) -> None:
        self.orig_img = orig_img
        self.expl_func = expl_func
        self.img_path = img_path
        self.n_sample = args.n_sample
        self.alpha = args.alpha
        self.noise_sigma = args.sigma
        self.logger = Logger(args, self.img_path)
        self.pred_method = args.pred_method
        self.eval_method = args.eval_method
        self.data = args.data
        self.T = TransformFactory(self.logger)
        self.batch_size = args.batch_size
        self.base_entropy = 4.2894835

        if self.data == "imagenet":
            if args.orig_input_method == "center_crop_224":
                # self.temp_img = center_crop_224(resize_322(orig_img))
                self.temp_img = center_crop_224(resize_232(orig_img))
            else:
                self.temp_img = resize_224(orig_img)
        elif self.data == "cifar10":
            self.temp_img = center_crop_32(resize_46(orig_img))
    
        self.temp_img = imagenet_normalize(tensorize(self.temp_img))
        self.orig_expl, self.orig_probs, self.orig_pred = expl_func(self.temp_img.unsqueeze(0).cuda(), "init")

        self.upsample = args.upsample
        if self.upsample:
            self.orig_expl = center_crop_224(self.orig_expl)
        self.transform = args.transform
        self.sign = args.sign
        self.reduction = args.reduction
        self.run_option = args.run_option
        self.convert_device = args.convert_device

        if self.sign == "absolute":
            self.orig_expl = torch.abs(self.orig_expl)

        if self.orig_expl.shape[1] > 1 and self.reduction == 'sum':
            self.orig_expl = torch.sum(self.orig_expl, axis=1).unsqueeze(0)
        elif self.orig_expl.shape[1] > 1 and self.reduction == 'mean':
            self.orig_expl = torch.mean(self.orig_expl, axis=1).unsqueeze(0)

        print("Original expl shape: ", self.orig_expl.shape)

    def make_confidence_set(self, transform_configs=None):
        print("Make confidence set ...")
        true_expls = []
        T_spatial_configs = []
        scores = []
        t = 0
        n_try = 0
        pbar = tqdm(total = self.n_sample)
    
        while t < self.n_sample:
            img_batch = to_tensor(self.orig_img).repeat(self.batch_size, 1, 1, 1)
            transformed_img = self.T(img_batch).cuda()
            _true_expl, _probs, _target = self.expl_func(transformed_img, self.orig_pred)
            correct_indices = np.array(torch.where(_target == self.orig_pred)[0].detach().cpu())
            entropies = np.array(list(map(entropy, _probs.detach().cpu())))
            entropy_indices = np.where(entropies < self.base_entropy)[0]

            intersect_indices = np.intersect1d(correct_indices, entropy_indices)
            if len(intersect_indices) == 0:
                n_try += 1
                if n_try == 5:
                    self.logger.log_long_time_file(self.img_path)
                    print("skipped!")
                    return
            else:
                t += len(intersect_indices)
                pbar.update(len(intersect_indices))
                self.logger.save_intersect_index(intersect_indices)

            true_expl = _true_expl[intersect_indices]

            if self.reduction == 'sum':
                true_expl = torch.sum(true_expl, axis = 1).unsqueeze(1)

            true_expls.append(true_expl.detach().cpu().numpy())


        true_expls = np.concatenate(true_expls)[:2000]
        print("True expl shape: ", true_expls.shape)

        self.logger.save_orig_true_config(self.orig_expl.detach().squeeze(0).cpu().numpy(), true_expls, T_spatial_configs)

        if self.run_option == 'all':
            self.q_hat = np.quantile(scores, np.ceil((scores.shape[0]+1) * (1-self.alpha)) / scores.shape[0], axis = 0)

        pbar.close()


    def evaluate(self):
        print("Evaluating ...")
        test_expls = []
        predictions = []

        pbar = tqdm(total = self.n_sample)
        if self.eval_method == "orig":
            t = 0
            while t < self.n_sample:
                T_spatial, T_inv_spatial = get_spatial_transform()
                T_color = get_color_transform()

                tensorized_img = tensorize(resize_322(self.orig_img))
                noise = self.noise_sigma * torch.randn_like(tensorized_img)

                if self.transform == "both":
                    transformed_img = imagenet_normalize(tensorize(T_spatial(T_color(resize_322(self.orig_img)))) - noise)
                elif self.transform == "spatial":
                    transformed_img = T_spatial(imagenet_normalize(tensorized_img - noise))
                transformed_img = transformed_img.unsqueeze(0).cuda()


                _transformed_expl, _target = self.expl_func(transformed_img, self.orig_pred)

                if _target != self.orig_pred:
                    continue
                else:
                    t += 1
                    pbar.update(1)
                    predictions.append(_target.item())
                
                transformed_expl = center_crop_224(T_inv_spatial(_transformed_expl))

                if self.convert_device == False and self.reduction == 'sum':
                    transformed_expl = torch.sum(transformed_expl, axis = 1).unsqueeze(1)

                transformed_expl = transformed_expl.detach().squeeze(0).cpu().numpy()


                if self.convert_device:
                    if self.sign == "absolute":
                        transformed_expl = np.abs(transformed_expl)

                    if self.reduction == 'sum':
                        transformed_expl = np.expand_dims(np.sum(transformed_expl, axis = 0), axis = 0)
                    elif self.reduction == 'mean':
                        transformed_expl = np.expand_dims(np.mean(transformed_expl, axis = 0), axis = 0)

                test_expls.append(transformed_expl)        

        
            assert sum(predictions) == self.orig_pred.item() * self.n_sample
            test_expls = np.stack(test_expls)

            print("Test expl shape: ", test_expls.shape)
            self.logger.save_test(test_expls)

            if self.run_option == "all":
                self.conf_low, self.conf_high = self.get_conf_interval(self.orig_expl.detach().squeeze(0).cpu().numpy(), self.q_hat)
                self.coverage_map = self.calc_coverage_prob(test_expls, self.conf_low, self.conf_high)

            pbar.close()
        
        elif self.eval_method == "new":
            raise NotImplementedError("code under construction")
            conf_lows = []
            conf_highs = []
            for _ in tqdm(range(self.n_sample)):
                T_spatial, T_inv_spatial = get_spatial_transform()
                T_color = get_color_transform()

                tensorized_img = tensorize(resize_322(self.orig_img))
                noise = self.noise_sigma * torch.randn_like(tensorized_img)

                if self.transform == "both":
                    transformed_img = T_spatial(imagenet_normalize(T_color(tensorized_img - noise)))
                elif self.transform == "spatial":
                    transformed_img = T_spatial(imagenet_normalize(tensorized_img - noise))
                transformed_img = transformed_img.unsqueeze(0).cuda()

                transformed_expl = center_crop_224(T_inv_spatial(self.expl_func(transformed_img)))
                transformed_expl = transformed_expl.detach().squeeze(0).cpu().numpy()

                if self.pred_method == "new":
                    if self.transform == "both":
                        pred_expl = center_crop_224(self.expl_func(imagenet_normalize(T_color(tensorized_img)).unsqueeze(0).cuda())) 
                    elif self.transform == "spatial":
                        pred_expl = center_crop_224(self.expl_func(imagenet_normalize(tensorized_img).unsqueeze(0).cuda()))
                elif self.pred_method == "orig":
                    pred_expl = center_crop_224(self.expl_func(imagenet_normalize(tensorized_img).unsqueeze(0).cuda()))

                pred_expl = pred_expl.detach().squeeze(0).cpu().numpy()

                test_expls.append(transformed_expl)
                conf_low, conf_high = self.get_conf_interval(pred_expl, self.q_hat)
                conf_lows.append(conf_low)
                conf_highs.append(conf_high)

            test_expls = np.stack(test_expls)
            self.conf_lows = np.stack(conf_lows)
            self.conf_highs = np.stack(conf_highs)

            self.coverage_map = self.calc_coverage_prob(test_expls, self.conf_lows, self.conf_highs)
            

    def logging(self):
        if self.eval_method == "orig":
            self.logger.save_conf_interval(self.orig_expl.detach().squeeze(0).cpu().numpy(), self.conf_low, self.conf_high)
            self.logger.save_histogram(self.coverage_map)
            self.logger.save_coverage_map(self.coverage_map, self.conf_low, self.conf_high)
        elif self.eval_method == "new":
            self.logger.save_conf_interval(self.orig_expl.detach().squeeze(0).cpu().numpy(), self.conf_lows, self.conf_highs)
            self.logger.save_histogram(self.coverage_map)
            self.logger.save_coverage_map(self.coverage_map)


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
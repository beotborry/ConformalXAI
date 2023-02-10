import torch
import numpy as np
import matplotlib.pyplot as plt
from transform_factory import resize_322, center_crop_224, tensorize, get_spatial_transform, get_color_transform, imagenet_normalize, PIL2Tensor, ToPIL, gauss_noise_tensor, resize_232, resize_224
from logger import Logger
from tqdm import tqdm
import time

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


        self.temp_img = center_crop_224(resize_322(orig_img))
        self.temp_img = imagenet_normalize(tensorize(self.temp_img))
        self.orig_expl, self.orig_pred = expl_func(self.temp_img.unsqueeze(0).cuda(), "init")

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

    def make_confidence_set(self):
        print("Make confidence set ...")
        true_expls = []
        T_spatial_configs = []
        scores = []
        t = 0
        pbar = tqdm(total = self.n_sample)
        while t < self.n_sample:
            T_spatial, T_inv_spatial, T_spatial_config = get_spatial_transform()
            T_color = get_color_transform()

            if self.transform == "both":
                tmp = ToPIL(gauss_noise_tensor(PIL2Tensor(resize_322(self.orig_img))))
                transformed_img = imagenet_normalize(tensorize(resize_224(T_spatial(T_color(tmp)))))

            transformed_img = transformed_img.unsqueeze(0).cuda()

            _true_expl, _target = self.expl_func(transformed_img, self.orig_pred)
            
            if _target != self.orig_pred:
                continue
            else:
                t += 1
                pbar.update(1)
                T_spatial_configs.append(T_spatial_config)
                    
            # print(_true_expl.shape) # (1, 1, 322, 322)
            
            if self.upsample:
                true_expl = center_crop_224(T_inv_spatial(_true_expl))
            else:
                true_expl = _true_expl # FIXME

            if self.reduction == 'sum':
                true_expl = torch.sum(true_expl, axis = 1).unsqueeze(1)

            
            assert true_expl.shape == self.orig_expl.shape

            true_expls.append(true_expl.detach().squeeze(0).cpu().numpy())


        true_expls = np.stack(true_expls)
        T_spatial_configs = np.stack(T_spatial_configs)
        print("True expl shape: ", true_expls.shape)
        print("Config shape: ", T_spatial_configs.shape)

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
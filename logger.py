import numpy as np
import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, args, img_path) -> None:
        self.args = args
        self.img_path = img_path
        self.base_logname = self.generate_base_logname()
        self.save_path = self.generate_save_path()

    def generate_save_path(self):
        save_path = f"./results/{self.args.date}_dataset_{self.args.dataset}_orig_input_method_{self.args.orig_input_method}_pred_{self.args.pred_method}_eval_{self.args.eval_method}_transform_{self.args.transform}_sign_{self.args.sign}_reduction_{self.args.reduction}"

        if os.path.exists(save_path) == False:
            os.mkdir(save_path)

        return save_path

    def generate_base_logname(self):
        img_name = os.path.basename(self.img_path)

        return f"{img_name}_expl_{self.args.expl_method}_sample_{self.args.n_sample}_sigma_{self.args.sigma}_seed_{self.args.seed}"

        # return f"{img_name}_expl_{self.args.expl_method}_alpha_{self.args.alpha}_sample_{self.args.n_sample}_sigma_{self.args.sigma}_seed_{self.args.seed}"

    def save_conf_interval(self, orig_expl, conf_low, conf_high):
        with open(f"{self.save_path}/{self.base_logname}.npy", "wb") as f:
            np.save(f, orig_expl)
            np.save(f, conf_low)
            np.save(f, conf_high)

    def save_histogram(self, coverage_map):
        plt.hist(coverage_map.flatten())
        plt.xticks(np.arange(0.2, 1.0, 0.05))
        plt.title(f'{self.args.expl_method}')
        plt.savefig(f"{self.save_path}/{self.base_logname}_hist.jpg")
        plt.clf()
        
    def save_coverage_map(self, coverage_map, conf_low = None, conf_high = None):
        plt.imshow(coverage_map.mean(axis = 0), cmap='hot', interpolation='nearest')

        if conf_low is not None and conf_high is not None:
            plt.text(20, 220, f"avg length: {(conf_high - conf_low).flatten().mean():.3f}")
        plt.text(180, 200, f"mean: {coverage_map.squeeze().flatten().mean():.3f}")
        plt.text(180, 180, f"max: {coverage_map.squeeze().flatten().max():.3f}")
        plt.text(180, 220, f"min: {coverage_map.squeeze().flatten().min():.3f}")
        plt.colorbar()
        plt.title(f'{self.args.expl_method}')
        plt.savefig(f"{self.save_path}/{self.base_logname}_coverage_map.jpg")
        plt.clf()
    
    def save_orig_true_pred(self, orig, true, pred):
        with open(f"{self.save_path}/{self.base_logname}_orig_true_pred.npy", "wb") as f:
            np.save(f, orig)
            np.save(f, true)
            np.save(f, pred)

    def save_test(self, test):
        with open(f"{self.save_path}/{self.base_logname}_test.npy", "wb") as f:
            np.save(f, test)

    def save_orig_true_score(self, orig, true, score):
        with open(f"{self.save_path}/{self.base_logname}_orig_score.npy", "wb") as f:
            np.save(f, orig)
            np.save(f, true)
            np.save(f, score)

    def save_orig_true_config(self, orig, true, config):
        with open(f"{self.save_path}/{self.base_logname}_orig_true_config.npy", "wb") as f:
            np.save(f, orig)
            np.save(f, true)
            np.save(f, config)
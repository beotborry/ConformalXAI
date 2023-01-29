import numpy as np
import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, args) -> None:
        self.args = args
        self.base_logname = self.generate_base_logname()

    def generate_base_logname(self):
        img_name = os.path.basename(self.args.img_path)

        return f"{img_name}_expl_{self.args.expl_method}_alpha_{self.args.alpha}_sample_{self.args.n_sample}"

    def save_conf_interval(self, orig_expl, conf_low, conf_high):
        with open(f"./{self.base_logname}.npy", "wb") as f:
            np.save(f, orig_expl)
            np.save(f, conf_low)
            np.save(f, conf_high)

    def save_histogram(self, coverage_map):
        plt.hist(coverage_map.flatten())
        plt.xticks(np.arange(0.2, 1.0, 0.05))
        plt.savefig(f"./{self.base_logname}_hist.jpg")
        plt.clf()
        
    def save_coverage_map(self, coverage_map, conf_low, conf_high):
        plt.imshow(coverage_map.mean(axis = 0), cmap='hot', interpolation='nearest')
        plt.text(20, 220, f"avg length: {(conf_high - conf_low).flatten().mean():.3f}")
        plt.text(180, 200, f"mean: {coverage_map.squeeze().flatten().mean():.3f}")
        plt.text(180, 180, f"max: {coverage_map.squeeze().flatten().max():.3f}")
        plt.text(180, 220, f"min: {coverage_map.squeeze().flatten().min():.3f}")
        plt.colorbar()
        plt.savefig(f"./{self.base_logname}_coverage_map.jpg")
        plt.clf()
    
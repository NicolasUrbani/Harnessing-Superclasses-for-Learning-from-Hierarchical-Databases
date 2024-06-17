import math

import hydra
import numpy as np
import omegaconf
from omegaconf import DictConfig
from tqdm import tqdm


# Adapted from https://www.johndcook.com/blog/standard_deviation/
class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    # Push values by batch
    def push_array(self, x):
        self.n += len(x)

        if self.n == 1:
            self.old_m = self.new_m = np.mean(x)
            self.old_s = 0
        else:
            self.new_m = self.old_m + (np.sum(x) - len(x) * self.old_m) / self.n
            self.new_s = (
                self.old_s
                + (self.n - len(x)) * (self.old_m - self.new_m) ** 2
                + np.sum((x - self.new_m) ** 2)
            )

            self.old_m = self.new_m
            self.old_s = self.new_s

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def var(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return math.sqrt(self.var())


@hydra.main(config_path="../configs", version_base=None, config_name="dataset_stats.yaml")
def main(cfg: DictConfig):
    print(omegaconf.OmegaConf.to_yaml(cfg))

    rs0 = RunningStats()
    rs1 = RunningStats()
    rs2 = RunningStats()

    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    for image, target in tqdm(datamodule.train_dataloader()):
        rs0.push_array(image[:, 0, :].flatten().numpy())
        rs1.push_array(image[:, 1, :].flatten().numpy())
        rs2.push_array(image[:, 2, :].flatten().numpy())

    print(f"Stats for {datamodule.datasets.name}")
    print(f"mean: ({rs0.mean()}, {rs1.mean()}, {rs2.mean()})")
    print(f"std: ({rs0.std()}, {rs1.std()}, {rs2.std()})")


if __name__ == '__main__':
    main()

import subprocess
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="rsync.yaml")
def main(cfg: DictConfig) -> None:
    print("Executing:", cfg.rsync_command)
    process = subprocess.run(cfg.rsync_command, shell=True)


if __name__ == "__main__":
    main()

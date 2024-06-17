import sys
import shlex

import hydra
import omegaconf
from omegaconf import DictConfig
from lightning.pytorch.utilities.rank_zero import rank_zero_only


from .utils import extract_hyperparameters


@hydra.main(config_path="../configs", version_base=None, config_name="main.yaml")
def main(cfg: DictConfig):
    if rank_zero_only.rank == 0:
        print(omegaconf.OmegaConf.to_yaml(cfg))

    model = hydra.utils.instantiate(cfg.model.lightning_model, cfg)
    datamodule = hydra.utils.instantiate(cfg.dataset)
    logger = hydra.utils.instantiate(cfg.logger.target)

    # Log description and hyperparameters from configuration
    logger.experiment.set_tag(logger.run_id, "mlflow.note.content", cfg.description)
    logger.log_hyperparams(extract_hyperparameters(cfg))

    # Log command line
    command_line = "python " + " ".join(map(shlex.quote, sys.argv))
    logger.experiment.log_param(logger.run_id, "command", command_line)

    # Log hydra configuration
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.experiment.log_artifacts(logger.run_id, output_dir)

    trainer = hydra.utils.instantiate(cfg.trainer.target, logger=logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()

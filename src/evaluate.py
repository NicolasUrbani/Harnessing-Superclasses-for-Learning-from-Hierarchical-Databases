import copy
import os

import hydra
import lightning as L
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from torchmetrics import CatMetric, MetricCollection

from .metrics import HierarchicalAccuracy, WassersteinTreeDistance
from .model import LightningModel
from .tree import add_weight_to_nodes
from .utils import extract_hyperparameters
from .utils_mlflow import Run


class Wrapper(L.LightningModule):
    """Wrapper around of Lightning model to compute metrics."""

    # Template file name to store metrics
    template = "{run_name}.pth"

    def __init__(self, lightning_model, *, run_name, metric_kwargs):
        super().__init__()

        self.model = lightning_model
        self.run_name = run_name

        self.metric_kwargs = metric_kwargs

        def make_tree(tree, **kwargs):
            tree = copy.deepcopy(tree)
            name = kwargs.pop("name")
            add_weight_to_nodes(tree, **kwargs)
            return name, tree

        tree_names = [
            make_tree(self.model.tree, **kwargs)
            for kwargs in self.metric_kwargs
        ]

        self.wasserstein_dists = MetricCollection(
            {
                name: WassersteinTreeDistance(tree, average="samplewise")
                for name, tree in tree_names
            }
        )

        self.wasserstein_dists_per_class = MetricCollection(
            {
                name: WassersteinTreeDistance(tree, average="none")
                for name, tree in tree_names
            }
        )

        self.hierarchical_accuracy = MetricCollection(
            {name: HierarchicalAccuracy(tree) for name, tree in tree_names}
        )

        self.klass = CatMetric()

    def forward(self, x):
        return self.model.forward(x)

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        preds = torch.argmax(output, dim=1)
        softmax_output = F.softmax(output, dim=1)
        self.wasserstein_dists(softmax_output, target)
        self.wasserstein_dists_per_class(softmax_output, target)
        self.hierarchical_accuracy(preds, target)
        self.klass(target)

    def on_validation_end(self):
        data = {
            "wasserstein_dists": self.wasserstein_dists.compute(),
            "class": self.klass.compute(),
            "wasserstein_dists_per_class": self.wasserstein_dists_per_class.compute(),
            "hierarchical_accuracy": self.hierarchical_accuracy.compute(),
        }

        if rank_zero_only.rank == 0:
            filename = self.template.format(run_name=self.run_name)

            torch.save(data, filename)
            self.logger.experiment.log_artifact(self.logger.run_id, filename)
            os.remove(filename)


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg):
    """
    Calcul des distances de Wasserstein pour chaque exemple.
    """

    # Setup logger for all runs
    model_runs = cfg.model_runs.split(",")
    logger = hydra.utils.instantiate(cfg.logger.target)
    experiment_name = cfg.experiment_name
    logger.log_hyperparams(extract_hyperparameters(cfg, max_depth=1))

    # Reset hydra config to reload a config from a saved artifact
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    links = []
    for run_name in model_runs:
        run = Run(cfg, run_name=run_name)
        links.append(run.link)

        with initialize(version_base=None, config_path="../configs"):
            # Load the overrides of that run
            output_dir = run.get_hydra_config_path()
            original_overrides = OmegaConf.load(os.path.join(output_dir, "overrides.yaml"))

            # Override old experiment name
            original_overrides.append(f"experiment_name={experiment_name}")

            # Combine current config with overrides that were used for that run
            cfg_model = compose(config_name="main", overrides=original_overrides)

            # Now that the configuration is correct, we can load the model and
            # validate.
            datamodule = hydra.utils.instantiate(cfg_model.dataset)
            trainer = hydra.utils.instantiate(cfg_model.trainer.target, logger=logger)

            checkpoint = LightningModel.load_from_checkpoint(
                run.model_path, cfg=cfg_model
            )

            metric_kwargs = [
                dict(method="balanced", name="balanced"),
                dict(method="n_leaves", name="n_leaves"),
                dict(method="power", decay=0.5, name="power[0d5]"),
            ]

            model = Wrapper(
                checkpoint, run_name=run_name, metric_kwargs=metric_kwargs
            )
            trainer.validate(model, datamodule=datamodule)

    description = cfg.description.format(
        description=main.__doc__.strip(),
        model_runs=", ".join(links)
    )
    logger.experiment.set_tag(logger.run_id, "mlflow.note.content", description)


if __name__ == "__main__":
    main()

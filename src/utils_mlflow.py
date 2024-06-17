import os
import re
from pathlib import Path

import mlflow
import torch


class Run:
    """Wrapper around `Run` Mlflow object.

    Easy creation of run by name and convenient properties.
    """

    def __init__(self, cfg, experiment_name=None, run_name=None):
        self.cfg = cfg
        mlflow.set_tracking_uri(cfg.logger.tracking_uri)
        self.experiment_name = experiment_name
        self.run_name = run_name

        self._experiment = None
        self._run = None

    def get_hydra_config_path(self):
        config = self.path / ".hydra"
        if not config.exists():
            raise Exception("Hydra config path not found")
        return str(config)

    def __getattr__(self, name):
        if name.endswith("_run"):
            # Return corresponding run
            return Run(self.cfg, run_name=self.data.params[name])
        else:
            return self.data.params[name]

    def _set_experiment_and_run(self):
        if self.run_name is None:
            raise Exception("At least `run_name` must be specified")

        if self.experiment_name is not None:
            self._experiment = mlflow.get_experiment_by_name(self.experiment_name)

            run_ids = mlflow.search_runs(
                experiment_ids=self.experiment_name,
                filter_string=f'tags."mlflow.runName" LIKE "{self.run_name}"',
            )["run_id"]

            if run_ids is None or len(run_ids) == 0:
                raise Exception(
                    f"Run named {self.run_name} in experiment `{self.experiment_name}` not found"
                )

            self._run = mlflow.get_run(run_ids[0])

        elif self.experiment_name is None:
            run_ids = mlflow.search_runs(
                search_all_experiments=True,
                filter_string=f'tags."mlflow.runName" LIKE "{self.run_name}"',
            )["run_id"]
            if run_ids is None or len(run_ids) == 0:
                raise Exception(f"Run named `{self.run_name}` not found")

            self._run = mlflow.get_run(run_ids[0])
            self._experiment = mlflow.get_experiment(self._run.info.experiment_id)

    @property
    def experiment(self):
        if self._experiment is None:
            self._set_experiment_and_run()
        return self._experiment

    @property
    def run(self):
        if self._run is None:
            self._set_experiment_and_run()
        return self._run

    @property
    def run_id(self):
        return self.run.info.run_id

    @property
    def data(self):
        return self.run.data

    @property
    def path(self):
        """Path of artifacts directory for this run."""

        artifact_uri = self.run.info.artifact_uri

        # Translate path to local machine
        artifact_uri = artifact_uri.replace(
            self.cfg.paths.remote_artifact_location, self.cfg.paths.artifact_location
        )
        return Path(artifact_uri)

    @property
    def model_path(self):
        """Return the model found under `self.path`."""
        filepaths = list(
            list_files(Path(self.path) / "model" / "checkpoints", glob="*.ckpt")
        )
        if len(filepaths) != 1:
            raise Exception("No or more than one model")
        return str(filepaths[0])

    def model_paths(self, regex=None):
        """Generate models found under `self.path`.

        Models are files whose extension is .cpkt anywhere under `self.path`. A
        regex can be given to filter models.

        """

        yield from list_files(
            Path(self.path) / "model" / "checkpoints",
            regex=regex,
            glob="*.cpkt",
        )

    def artifact_paths(self, regex=None):
        yield from list_files(self.path, regex)

    def artifact(self, regex=None):
        artifacts = list(self.artifact_paths(regex=regex))
        if len(artifacts) != 1:
            raise Exception("No or more than one artifact found in `{self.path}`")
        return artifacts[0]

    def load_artifact(self, regex=None):
        artifacts = list(self.artifact_paths(regex=regex))
        if len(artifacts) != 1:
            raise Exception("")

        return torch.load(artifacts[0], map_location=torch.device("cpu"))

    @property
    def link(self):
        return f"[{self.run_name}](#/experiments/{self.experiment.experiment_id}/runs/{self.run_id})"

    @property
    def absolute_link(self):
        return f"{self.cfg.tracking_uri}/#/experiments/{self.experiment.experiment_id}/runs/{self.run_id}"


def list_files(path, regex=None, glob="*"):
    """Generate paths or tuples of paths and capturing groupes of files under `path`.

    Look for each file under `path` according to `glob` and whose name is
    matching `regex`. If the regex contains capturing groups, return a tuple of
    the file path and the capturing groups. Otherwise just return the file path.

    """

    def parse_group(group):
        try:
            return int(group)
        except:
            try:
                return float(group)
            except:
                return group

    for p in Path(path).rglob(glob):
        if regex is None:
            yield str(p)
        else:
            if (m := re.match(regex, p.name)) is not None:
                if m.groups():
                    args = [parse_group(arg) for arg in m.groups()]
                    yield str(p), *args
                else:
                    yield str(p)

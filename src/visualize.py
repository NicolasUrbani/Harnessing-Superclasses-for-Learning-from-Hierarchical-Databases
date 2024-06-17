import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf

from .utils_mlflow import Run


@hydra.main(version_base=None, config_path="../configs", config_name="visualize.yaml")
def main(cfg: DictConfig) -> None:
    func = globals()[cfg.entrypoint]
    func(cfg)


def vis0(cfg: DictConfig) -> None:
    """
    Histogrammes des distances de Wasserstein par modèles.
    """

    base_run = Run(cfg, run_name="awesome-midge-639")

    def gen_data():
        for run_name in base_run.model_runs.split(","):
            run = Run(cfg, run_name=run_name)
            ident = run.loss

            filename = base_run.artifact(regex=run_name)
            print(filename)
            data = torch.load(filename, map_location=torch.device("cpu"))

            for data in data["wasserstein_dists"]:
                yield ident, data.item()

    df = pd.DataFrame(gen_data(), columns=["run_name", "value"])
    print(df.info())
    sns.displot(x="value", hue="run_name", data=df, col="run_name")
    plt.show()


def vis01(cfg: DictConfig) -> None:
    """
    Fonction de répartition des distances de Wasserstein par modèles.
    """

    base_run = Run(cfg, run_name="awesome-midge-639")

    def gen_data():
        for run_name in base_run.model_runs.split(","):
            run = Run(cfg, run_name=run_name)
            ident = run.loss

            filename = base_run.artifact(regex=run_name)
            print(filename)
            data = torch.load(filename, map_location=torch.device("cpu"))

            for data in data["wasserstein_dists"]:
                yield ident, data.item()

    df = pd.DataFrame(gen_data(), columns=["run_name", "value"])
    print(df.info())
    sns.ecdfplot(x="value", hue="run_name", data=df)
    plt.show()



def vis1(cfg: DictConfig) -> None:
    """
    Histogrammes des distances de Wasserstein par classes et par modèles.
    """

    base_run = Run(cfg, run_name="awesome-midge-639")

    def gen_data():
        for run_name in base_run.model_runs.split(","):
            run = Run(cfg, run_name=run_name)
            ident = run.loss

            filename = base_run.artifact(regex=run_name)
            print(filename)
            data = torch.load(filename, map_location=torch.device("cpu"))

            for data, klass in zip(data["wasserstein_dists"], data["class"]):
                if klass > 5:
                    continue
                yield ident, data.item(), klass.item()

    df = pd.DataFrame(gen_data(), columns=["run_name", "value", "klass"])
    print(df.info())
    sns.displot(x="value", hue="run_name", data=df, col="run_name", row="klass")
    plt.show()


def vis2(cfg: DictConfig) -> None:
    """
    Distance de Wasserstein par classes et par modèles.
    """

    base_run = Run(cfg, run_name="awesome-midge-639")

    def gen_data():
        for run_name in base_run.model_runs.split(","):
            run = Run(cfg, run_name=run_name)
            ident = run.loss

            if ident == "hie_loss[power]":
                continue

            filename = base_run.artifact(regex=run_name)
            print(filename)
            data = torch.load(filename, map_location=torch.device("cpu"))

            for klass, data in enumerate(data["wasserstein_dists_per_class"]):
                if klass > 20:
                    continue
                yield ident, data.item(), klass

    df = pd.DataFrame(gen_data(), columns=["run_name", "value", "klass"])
    print(df.info())

    sns.catplot(x="klass", y="value", hue="run_name", data=df, kind="bar")
    plt.show()


def vis3(cfg: DictConfig) -> None:
    """

    """

    from .utils_vis import column_reorder
    base_run = Run(cfg, run_name="awesome-midge-639")

    def gen_data():
        a = []
        runs = []
        for run_name in base_run.model_runs.split(","):
            run = Run(cfg, run_name=run_name)
            ident = run.loss

            if ident not in ["hie_loss[n_leaves]", "cross-entropy"]:
                continue

            filename = base_run.artifact(regex=run_name)
            print(filename)
            data = torch.load(filename, map_location=torch.device("cpu"))

            runs.append(ident)
            a.append(data["wasserstein_dists_per_class"].numpy())

        a = np.array(a)
        a = column_reorder(a)

        for run_name, b in zip(runs, a):
            for i, a in enumerate(b):
                yield run_name, a.item(), i


    df = pd.DataFrame(gen_data(), columns=["run_name", "value", "klass"])
    print(df.info())

    sns.relplot(data=df, kind="line", x="klass", y="value", hue="run_name")
    plt.show()


RUN_NAME_RX = r"\w+-\w+-\d+"

def vis4(cfg: DictConfig) -> None:
    base_run = Run(cfg, run_name="bright-mare-663")

    def gen_data():
        a = []
        runs = []
        for filename, run_name, eval_method in base_run.artifact_paths(regex=rf"({RUN_NAME_RX})_(\w+).pth"):
            run = Run(cfg, run_name=run_name)
            ident = run.loss

            data = torch.load(filename, map_location=torch.device("cpu"))

            for data in data["wasserstein_dists"]:
                yield ident, data.item(), eval_method


    df = pd.DataFrame(gen_data(), columns=["train_loss", "wasserstein_dist", "eval_method"])
    print(df.info())

    sns.displot(data=df, kind="ecdf", x="wasserstein_dist", hue="train_loss", row="eval_method")
    plt.show()


if __name__ == '__main__':
def vis6(cfg: DictConfig):
    base_runs = ["angry-shrimp-372"]

    def gen_data():
        for base_run in base_runs:
            base_run = Run(cfg, run_name=base_run)

            for filename, run_name in base_run.artifact_paths(regex=rf"({RUN_NAME_RX}).pth"):
                data = torch.load(filename, map_location=torch.device("cpu"))

                run = Run(cfg, run_name=run_name)
                ident = run.loss

                for eval_method, acc in data["hierarchical_accuracy"].items():

                    yield ident, acc.item(), eval_method


    df = pd.DataFrame(
        gen_data(), columns=["train_loss", "hierarchical_acc", "eval_method"]
    )
    print(df.info())

    import plotly.express as px

    fig = px.bar(df, x="train_loss", y="hierarchical_acc", facet_row="eval_method")

    fig.show()

if __name__ == "__main__":
    main()

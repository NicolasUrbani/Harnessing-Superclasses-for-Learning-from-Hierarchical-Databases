import click
from omegaconf import OmegaConf, DictConfig
from omegaconf.resolvers import oc
from collections.abc import MutableMapping


def resolve_click(param_name):
    """Return a parameter taken from Click."""

    click_ctx = click.get_current_context(silent=True)
    if click_ctx is not None:
        return click_ctx.params[param_name]
    else:
        raise Exception(f"Unable to resolve param `{param_name}` from Click")


OmegaConf.register_new_resolver("click", resolve_click)


def resolve_decode_csv(csv, _parent_, _node_, _root_):
    """Return a list of Python objects using `oc.decode`."""

    return [oc.decode(e, _parent_, _node_) for e in csv.split(",")]


OmegaConf.register_new_resolver("decode_csv", resolve_decode_csv)


def resolve_greater_1(seq, yes, no):
    return yes if len(seq) > 1 else no


OmegaConf.register_new_resolver("greater_1", resolve_greater_1)


def extract_hyperparameters(cfg: DictConfig, max_depth=None):
    def extract_hyperparameters_aux(cfg: DictConfig, parent_key: str, current_depth: int):
        if max_depth is not None and current_depth >= max_depth:
            return []
        items = []
        if "hyperparameters" in cfg:
            items.extend([(k, v) for k, v in cfg["hyperparameters"].items() if v is not None])
        for key, value in cfg.items():
            if isinstance(value, MutableMapping):
                items.extend(extract_hyperparameters_aux(value, key, current_depth+1))
        return items

    return dict(extract_hyperparameters_aux(cfg, "", 0))

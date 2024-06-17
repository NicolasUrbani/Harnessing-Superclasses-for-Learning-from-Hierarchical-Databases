import platform
import re
import subprocess
from pathlib import Path

import hydra
from omegaconf import OmegaConf


def register_resolver(name):
    """Decorator to register a function as a resolver named `name`."""

    def decorator(func):
        OmegaConf.register_new_resolver(name, func)
        return func
    return decorator


def probe():
    """Return a list of indexes of available gpus.

    An available gpu is one that does not run any process.

    """

    # Get list of busy gpus serial number (ones that have at least one process)
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_serial", "--format=csv,noheader"],
        encoding="utf-8",
    )
    busy_gpu_serial = output.split("\n")

    # Get index to serial dictionary
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,serial", "--format=csv,noheader"],
        encoding="utf-8",
    )
    index_serial = {
        int(m.group(1)): m.group(2) for m in re.finditer("([0-9]), ([0-9]+)", output)
    }

    # Get indexes of gpus that are not busy
    available_index = [
        i for i, serial in index_serial.items() if serial not in busy_gpu_serial
    ]

    return available_index


@register_resolver("gpus_index")
def gpus_index(number):
    available = probe()
    if len(available) >= number:
        return available[:number]
    else:
        raise Exception(f"Not enough gpus: {number} requested, {len(available)} available")


@register_resolver("machine_name")
def machine_name():
    if platform.node().startswith("r") or "jean" in platform.node():
        return "jeanzay"
    else:
        return platform.node()


@register_resolver("attr")
def attr(path):
    return hydra.utils.get_object(path)


@register_resolver("callable")
def get_callable(path):
    return hydra.utils.get_method(path)


@register_resolver("failif")
def failif(a, b):
    if a != b:
        raise Exception("Not equal")


@register_resolver("unless_null")
def unless_null(a, b):
    return "" if a is None else b


@register_resolver("project_root")
def project_root():
    return str(Path(__file__).parent.parent.absolute())

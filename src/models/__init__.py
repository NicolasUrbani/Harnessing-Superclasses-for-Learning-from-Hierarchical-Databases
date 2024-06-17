import importlib


def get_function(module_name, function):
    try:
        module = importlib.import_module("." + module_name, package=__name__)
    except ModuleNotFoundError as e:
        raise Exception(f"Module {module_name} not found") from e

    try:
        func = getattr(module, function)
    except AttributeError as e:
        raise Exception(f"No function {function} found in module {module}")

    return func


def get_model(model_name, **kwargs):
    func = get_function(model_name, "model")
    return func(**kwargs)


__all__ = [
    "densenet_OOD",
    "resnet50_OOD"
]

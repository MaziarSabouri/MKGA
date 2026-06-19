from mkga.models.resnet34 import ResNet34
from mkga.models.resnet34_mkga import ResNet34_MKGA
from mkga.models.resnet34_resmkga import ResNet34_ResMKGA

__all__ = [
    "ResNet34",
    "ResNet34_MKGA",
    "ResNet34_ResMKGA",
    "SAM",
    "SAM_MKGA",
    "SAM_ResMKGA",
]

_SAM_MODELS = {
    "SAM": "mkga.models.sam",
    "SAM_MKGA": "mkga.models.sam_mkga",
    "SAM_ResMKGA": "mkga.models.sam_resmkga",
}


def __getattr__(name):
    if name in _SAM_MODELS:
        import importlib
        module = importlib.import_module(_SAM_MODELS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

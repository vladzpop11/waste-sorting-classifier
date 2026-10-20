"""Waste sorting image classifier."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("waste-sorting-classifier")
except PackageNotFoundError:  # running from a source checkout without installing (Kaggle)
    __version__ = "0+source"

"""Public package for Helix Support Intelligence."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("helix-support-intelligence")
except PackageNotFoundError:  # pragma: no cover - editable installs provide metadata
    __version__ = "0+unknown"

__all__ = ["__version__"]

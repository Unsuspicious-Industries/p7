"""Stable backend contracts for constrained generation.

Import the versioned module directly::

    from proposition7.backends.v1 import ConstraintBackend, BACKEND_API

Importing this package must not pull in Torch: provider7 and its tests depend
on the contract without depending on a model runtime.
"""

from . import v1

__all__ = ["v1"]

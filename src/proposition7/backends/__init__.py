"""Stable backend contracts for constrained generation.

Import the versioned module directly::

    from proposition7.backends.v1 import ConstraintBackend

Importing this package must not pull in Torch: a consumer may depend on the
contract without depending on a model runtime.
"""

from . import v1

__all__ = ["v1"]

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.pypeline.utils.registry import register_all_subclasses

# Trigger the import of the concrete backends so they register themselves.
from . import daemon_backend  # noqa: F401
from . import local_backend  # noqa: F401
from .backend import Backend, BackendFactory, BackendFeature, backend_factory_for

register_all_subclasses(BackendFactory)

__all__ = ["Backend", "BackendFactory", "BackendFeature", "backend_factory_for"]

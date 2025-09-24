#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
#
from revng.pypeline.utils.registry import register_all_subclasses

# Trigger the import of all modules so we can register the classes
from . import local_provider  # noqa: F401
from . import memory  # noqa: F401
from . import null  # noqa: F401
from .storage_provider import StorageProviderFactory

register_all_subclasses(StorageProviderFactory)

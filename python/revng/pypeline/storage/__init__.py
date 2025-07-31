#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import os
from typing import Optional

from revng.pypeline.model import Model
from revng.pypeline.utils.registry import get_singleton

from .memory import InMemoryStorageProvider
from .null import NullStorageProvider
from .sqlite3 import SQlite3StorageProvider
from .storage_provider import StorageProvider


def storage_provider_factory(
    model_path: str,
    storage_url: Optional[str] = None,
) -> StorageProvider:
    """
    Given a storage_url, return the appropriately instantiated StorageProvider.
    If storage_url is None, it defaults to "sqlite://pypeline.db".
    """
    storage_url = storage_url or "sqlite://pypeline.db"

    if storage_url.startswith("sqlite://"):
        # Create an empty file if it does not exist
        if not os.path.exists(model_path):
            model_ty: type[Model] = get_singleton(Model)
            empty_model = model_ty()
            with open(model_path, "wb") as f:
                f.write(empty_model.serialize())

        return SQlite3StorageProvider(
            db_path=storage_url[len("sqlite://") :],
            model_path=model_path,
        )
    elif storage_url.startswith("memory"):
        return InMemoryStorageProvider()
    elif storage_url.startswith("null"):
        return NullStorageProvider()
    else:
        raise ValueError(
            "Unknown storage provider `%s`. "
            "Please set the PYPELINE_STORAGE environment variable." % storage_url,
        )

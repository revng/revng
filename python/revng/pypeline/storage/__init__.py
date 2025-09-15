#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
from pathlib import Path
from typing import Optional

from revng.pypeline.model import Model
from revng.pypeline.utils.registry import get_singleton

from .memory import InMemoryStorageProvider
from .null import NullStorageProvider
from .sqlite3 import SQlite3StorageProvider
from .storage_provider import StorageProvider

logger = logging.getLogger(__name__)


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
        db_path = storage_url[len("sqlite://") :]
        resolved_model_path = Path(model_path)

        if not resolved_model_path.is_absolute():
            resolved_model_path = (Path(db_path).resolve().parent / resolved_model_path).resolve()

        # Create an empty file if it does not exist
        if not resolved_model_path.exists():
            model_ty: type[Model] = get_singleton(Model)  # type: ignore[type-abstract]
            empty_model = model_ty()
            with resolved_model_path.open("wb") as f:
                f.write(empty_model.serialize())

        logger.info(
            "Starting sqlite storage provider at %s with model %s", db_path, resolved_model_path
        )

        return SQlite3StorageProvider(
            db_path=str(db_path),
            model_path=str(resolved_model_path),
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

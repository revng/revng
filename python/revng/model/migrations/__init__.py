#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import importlib
from abc import ABC, abstractmethod

# mypy cannot see the generated code, so this "type: ignore" is necessary
from revng.model import Binary  # type: ignore

__all__ = ["migrate", "MigrationException", "MigrationBase"]


class MigrationException(Exception):
    pass


class MigrationBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def migrate(self, model: dict):
        pass


def migrate(model: dict):
    """Runs migrations procedures against `model` up to the schema version supported by the current
    version of revng.

    An input model with no `Version` is assumed to be in the latest schema version, and thus no
    migrations are run against it.
    """

    if "Version" not in model:
        return

    version = model["Version"]

    if version < 0:
        raise MigrationException(f"The Version must be a positive integer but is {version}")

    if version > Binary.SchemaVersion:
        raise MigrationException(
            f"The highest possible model schema version is {Binary.SchemaVersion}"
            + f"but the input model has version {version}"
        )

    for i in range(version + 1, Binary.SchemaVersion + 1):
        migration_libray = importlib.import_module(f".versions.v{i}", __package__)
        migration = migration_libray.Migration()
        migration.migrate(model)

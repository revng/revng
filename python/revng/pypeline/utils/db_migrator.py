#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
from abc import ABC, abstractmethod
from importlib.metadata import files as metadata_files
from pathlib import Path, PurePath


class DBMigrator(ABC):
    def __init__(self, package: str, path: PurePath):
        self._migrations: dict[int, Path] = {}

        files = metadata_files(package)
        assert files is not None
        for file in files:
            if not file.is_relative_to(path):
                continue

            match = re.match(r"v(?P<version>\d+)\.sql", file.name)
            assert match is not None
            absolute_path = file.locate()
            assert isinstance(absolute_path, Path)
            self._migrations[int(match["version"])] = absolute_path

        for value in range(1, len(self._migrations) + 1):
            assert value in self._migrations

        self.last_version = max(self._migrations)

    @abstractmethod
    def _create_tables_if_missing(self):
        """Create the migration table if missing"""

    @abstractmethod
    def _get_last_migration(self) -> int:
        """Get the last migration number present in the DB"""

    @abstractmethod
    def _apply_migration(self, version: int, body: str):
        """Apply the specified migration to the DB"""

    def migrate(self):
        self._create_tables_if_missing()
        db_version = self._get_last_migration()
        if self.last_version == db_version:
            return

        for version in range(db_version + 1, self.last_version + 1):
            self._apply_migration(version, self._migrations[version].read_text())

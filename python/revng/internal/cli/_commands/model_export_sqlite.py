#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options

_Loader = yaml.CSafeLoader
_Dumper = yaml.CSafeDumper


def log(message: str) -> None:
    print(message, file=sys.stderr)


TYPE_DEFINITION_REFERENCE_PATTERN = re.compile(r"/TypeDefinitions/(\d+)-(\w+)")


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS Platform (
            PlatformID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL UNIQUE,
            Architecture TEXT NOT NULL,
            OperatingSystem TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS Library (
            LibraryID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            PlatformID INTEGER NOT NULL,
            Header TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (PlatformID) REFERENCES Platform(PlatformID),
            UNIQUE(Name, PlatformID)
        );

        CREATE TABLE IF NOT EXISTS TypeDefinition (
            TypeDefinitionID INTEGER PRIMARY KEY AUTOINCREMENT,
            LibraryID INTEGER NOT NULL,
            Body TEXT NOT NULL,
            OriginalID INTEGER NOT NULL,
            FOREIGN KEY (LibraryID) REFERENCES Library(LibraryID)
        );

        CREATE TABLE IF NOT EXISTS Symbol (
            SymbolID INTEGER PRIMARY KEY AUTOINCREMENT,
            LibraryID INTEGER NOT NULL,
            Name TEXT NOT NULL,
            Kind TEXT NOT NULL,
            TypeDefinitionID INTEGER,
            FOREIGN KEY (LibraryID) REFERENCES Library(LibraryID),
            FOREIGN KEY (TypeDefinitionID) REFERENCES TypeDefinition(TypeDefinitionID)
        );

        CREATE TABLE IF NOT EXISTS TypeDefinitionDependencies (
            SourceTypeDefinitionID INTEGER NOT NULL,
            DestinationTypeDefinitionID INTEGER NOT NULL,
            PRIMARY KEY (SourceTypeDefinitionID, DestinationTypeDefinitionID),
            FOREIGN KEY (SourceTypeDefinitionID)
                REFERENCES TypeDefinition(TypeDefinitionID),
            FOREIGN KEY (DestinationTypeDefinitionID)
                REFERENCES TypeDefinition(TypeDefinitionID)
        );

        CREATE INDEX IF NOT EXISTS idx_platform_arch_os
            ON Platform(Architecture, OperatingSystem);
        CREATE INDEX IF NOT EXISTS idx_library_platform
            ON Library(PlatformID);
        CREATE INDEX IF NOT EXISTS idx_symbol_name ON Symbol(Name);
        CREATE INDEX IF NOT EXISTS idx_symbol_library ON Symbol(LibraryID);
        CREATE INDEX IF NOT EXISTS idx_deps_source
            ON TypeDefinitionDependencies(SourceTypeDefinitionID);
        CREATE INDEX IF NOT EXISTS idx_deps_dest
            ON TypeDefinitionDependencies(DestinationTypeDefinitionID);
    """
    )


def find_type_references(node: Any) -> set:
    references = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "Definition" and isinstance(value, str):
                match = TYPE_DEFINITION_REFERENCE_PATTERN.search(value)
                if match:
                    references.add(int(match.group(1)))
            else:
                references.update(find_type_references(value))
    elif isinstance(node, list):
        for item in node:
            references.update(find_type_references(item))
    return references


def import_model(
    connection: sqlite3.Connection,
    yaml_path: Path,
    platform_name: str,
    operating_system: str,
    library_name: str | None = None,
) -> None:
    with open(yaml_path) as yaml_file:
        model = yaml.load(yaml_file, Loader=_Loader)

    if not model:
        return

    if library_name is None:
        library_name = yaml_path.stem

    architecture = model.get("Architecture", "Invalid")

    header_fields = {
        key: value
        for key, value in model.items()
        if key not in ("ImportedDynamicFunctions", "TypeDefinitions")
    }
    header = yaml.dump(
        header_fields, Dumper=_Dumper, default_flow_style=False, allow_unicode=True, sort_keys=False
    )

    connection.execute(
        "INSERT OR IGNORE INTO Platform (Name, Architecture, OperatingSystem) VALUES (?, ?, ?)",
        (platform_name, architecture, operating_system),
    )
    platform_id = connection.execute(
        "SELECT PlatformID FROM Platform WHERE Name = ?", (platform_name,)
    ).fetchone()[0]

    connection.execute(
        "INSERT OR IGNORE INTO Library (Name, PlatformID, Header) VALUES (?, ?, ?)",
        (library_name, platform_id, header),
    )
    library_id = connection.execute(
        "SELECT LibraryID FROM Library WHERE Name = ? AND PlatformID = ?",
        (library_name, platform_id),
    ).fetchone()[0]

    original_id_to_database_id = {}
    type_definitions = model.get("TypeDefinitions", [])

    for type_definition in type_definitions:
        original_id = type_definition["ID"]
        body = yaml.dump(
            type_definition,
            Dumper=_Dumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        cursor = connection.execute(
            "INSERT INTO TypeDefinition (LibraryID, Body, OriginalID) VALUES (?, ?, ?)",
            (library_id, body, original_id),
        )
        original_id_to_database_id[original_id] = cursor.lastrowid

    for type_definition in type_definitions:
        database_id = original_id_to_database_id[type_definition["ID"]]
        for reference_id in find_type_references(type_definition) - {type_definition["ID"]}:
            if reference_id in original_id_to_database_id:
                connection.execute(
                    "INSERT OR IGNORE INTO TypeDefinitionDependencies "
                    "(SourceTypeDefinitionID, DestinationTypeDefinitionID) "
                    "VALUES (?, ?)",
                    (original_id_to_database_id[reference_id], database_id),
                )

    for function in model.get("ImportedDynamicFunctions", []):
        type_definition_database_id = None
        prototype = function.get("Prototype")
        if prototype and prototype.get("Kind") == "DefinedType":
            match = TYPE_DEFINITION_REFERENCE_PATTERN.search(prototype.get("Definition", ""))
            if match:
                type_definition_database_id = original_id_to_database_id.get(int(match.group(1)))
        connection.execute(
            "INSERT INTO Symbol (LibraryID, Name, Kind, TypeDefinitionID) " "VALUES (?, ?, ?, ?)",
            (
                library_id,
                function["Name"],
                "ImportedDynamicFunction",
                type_definition_database_id,
            ),
        )


class ModelExportSqliteCommand(Command):
    def __init__(self):
        super().__init__(
            ("model", "export", "sqlite"),
            "Import rev.ng YAML models into a SQLite database",
        )

    def register_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--db",
            required=True,
            help="Path to the output SQLite database",
        )
        parser.add_argument(
            "--platform",
            required=True,
            help="Platform name (e.g. ubuntu-24-04-x86-64)",
        )
        parser.add_argument(
            "--operating-system",
            required=True,
            help="Operating system (e.g. linux, windows, macos)",
        )
        parser.add_argument(
            "--library",
            default=None,
            help="Override library name (default: filename stem, "
            "or path relative to --prefix with .yml stripped)",
        )
        parser.add_argument(
            "--prefix",
            default=None,
            help="When set, library name defaults to the path of the YAML "
            "relative to this prefix, with the .yml suffix stripped. "
            "Ignored if --library is also set.",
        )
        parser.add_argument(
            "models",
            nargs="+",
            help="YAML model file(s) to import",
        )

    def run(self, options: Options) -> int:
        args = options.parsed_args

        connection = sqlite3.connect(args.db)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        create_schema(connection)

        prefix = Path(args.prefix).resolve() if args.prefix is not None else None

        for model_path in args.models:
            path = Path(model_path)
            log(f"Importing {path.name}...")
            library_name = args.library
            if library_name is None and prefix is not None:
                relative = path.resolve().relative_to(prefix)
                library_name = relative.with_suffix("").as_posix()
            import_model(connection, path, args.platform, args.operating_system, library_name)

        connection.commit()
        connection.close()
        log("Import complete.")
        return 0


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(ModelExportSqliteCommand())

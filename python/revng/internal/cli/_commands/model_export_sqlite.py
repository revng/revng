#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
import sqlite3
from pathlib import Path
from typing import Any

import click
import yaml

from revng.internal.cli.common import CommandRegistry, cli_logger

_Loader = yaml.CSafeLoader
_Dumper = yaml.CSafeDumper


TYPE_DEFINITION_REFERENCE_PATTERN = re.compile(r"/TypeDefinitions/(\d+)-(\w+)")

TRANSFERABLE_SYMBOL_FIELDS = ("Comment", "Attributes")


class _BodyDumper(yaml.SafeDumper):
    """Indent block sequences, so that a body is emitted as:

        Attributes:
          - NoReturn

    rather than the default:

        Attributes:
        - NoReturn

    Both parse the same, but the importer splices this into a larger document,
    where the un-indented form is harder to read.
    """

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def symbol_body(function: dict) -> str:
    """Serialize the part of a `{Dynamic,}Function` that has no dedicated
    `Symbol` column as a YAML mapping. Empty when there is nothing to store."""
    fields = {key: function[key] for key in TRANSFERABLE_SYMBOL_FIELDS if function.get(key)}
    if not fields:
        return ""

    return yaml.dump(
        fields,
        Dumper=_BodyDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


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
            Body TEXT NOT NULL DEFAULT '',
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


def assert_every_symbol_has_a_prototype(connection: sqlite3.Connection, stage: str) -> None:
    cursor = connection.execute("SELECT COUNT(*) FROM Symbol WHERE TypeDefinitionID IS NULL")
    (count,) = cursor.fetchone()
    if count != 0:
        raise RuntimeError(
            f"sanity check failed {stage}: {count} Symbol row(s) have no " f"prototype attached"
        )


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

    def resolve_prototype_id(function: dict) -> int | None:
        prototype = function.get("Prototype")
        if not prototype or prototype.get("Kind") != "DefinedType":
            return None
        match = TYPE_DEFINITION_REFERENCE_PATTERN.search(prototype.get("Definition", ""))
        if not match:
            return None
        return original_id_to_database_id.get(int(match.group(1)))

    symbol_rows: list[tuple[int, str, str, int | None, str]] = []

    for function in model.get("ImportedDynamicFunctions", []):
        type_definition_database_id = resolve_prototype_id(function)
        if type_definition_database_id is None:
            continue
        symbol_rows.append(
            (
                library_id,
                function["Name"],
                "ImportedDynamicFunction",
                type_definition_database_id,
                symbol_body(function),
            )
        )

    for function in model.get("Functions", []):
        type_definition_database_id = resolve_prototype_id(function)
        if type_definition_database_id is None:
            continue
        body = symbol_body(function)
        for name in function.get("ExportedNames") or []:
            symbol_rows.append((library_id, name, "Function", type_definition_database_id, body))

    if symbol_rows:
        connection.executemany(
            "INSERT INTO Symbol (LibraryID, Name, Kind, TypeDefinitionID, Body) "
            "VALUES (?, ?, ?, ?, ?)",
            symbol_rows,
        )


@click.command(name="sqlite", help="Import rev.ng YAML models into a SQLite database")
@click.option("--db", required=True, help="Path to the output SQLite database")
@click.option("--platform", required=True, help="Platform name (e.g. ubuntu-24-04-x86-64)")
@click.option(
    "--operating-system", required=True, help="Operating system (e.g. linux, windows, macos)"
)
@click.option(
    "--library",
    default=None,
    help="Override library name (default: filename stem, "
    "or path relative to --prefix with .yml stripped)",
)
@click.option(
    "--prefix",
    default=None,
    help="When set, library name defaults to the path of the YAML "
    "relative to this prefix, with the .yml suffix stripped. "
    "Ignored if --library is also set.",
)
@click.argument("models", metavar="MODELS...", nargs=-1, required=True)
def model_export_sqlite(
    db: str,
    platform: str,
    operating_system: str,
    library: str | None,
    prefix: str | None,
    models: tuple[str, ...],
) -> int:
    connection = sqlite3.connect(db)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    create_schema(connection)

    assert_every_symbol_has_a_prototype(connection, "before import")

    prefix_path = Path(prefix).resolve() if prefix is not None else None

    for model_path in models:
        path = Path(model_path)
        cli_logger.log(f"Importing {path.name}...")
        library_name = library
        if library_name is None and prefix_path is not None:
            relative = path.resolve().relative_to(prefix_path)
            library_name = relative.with_suffix("").as_posix()
        import_model(connection, path, platform, operating_system, library_name)

    connection.commit()
    assert_every_symbol_has_a_prototype(connection, "after import")
    connection.close()
    cli_logger.log("Import complete.")
    return 0


def setup(registry: CommandRegistry):
    registry.register(("model", "export"), model_export_sqlite)

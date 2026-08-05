#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import click
import jinja2
import jsonschema
import yaml

from revng.internal.cli.common import CommandRegistry
from revng.support import get_root

Change = (
    Tuple[Literal["definition"], Literal["add", "remove"], str]
    | Tuple[Literal["definition"], Literal["edit"], str, str]
    | Tuple[Literal["sub_definition"], Literal["add", "remove"], str, str]
    | Tuple[Literal["sub_definition"], Literal["edit"], str, str, str]
)


def _get_metaschema():
    metaschema_path = get_root() / "share/revng/tuple-tree-generator/metaschema.yml"

    with open(metaschema_path) as file:
        metaschema = yaml.safe_load(file)

    assert len(metaschema)

    return metaschema


def _get_raw_input_from_upstream(upstream_branch: str, schema_path: str) -> str:
    try:
        git_process = subprocess.run(
            ["git", "show", f"{upstream_branch}:{schema_path}"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Error while running git: {error.stderr.decode().strip()}") from error

    return git_process.stdout.decode()


def _restructure(schema: Dict) -> Dict:
    """Restructures the schema into a nested dictionaries where the `Name` field of types and
    fields/members are keys.
    """
    result: Dict[str, Any] = {}

    for definition in schema["definitions"]:
        definition_name = definition["name"]

        sub_definitions_list = (
            definition["members"] if definition["type"] == "enum" else definition["fields"]
        )

        sub_definitions = {}
        for sub_definition in sub_definitions_list:
            sub_definitions_name = sub_definition["name"]
            sub_definition_properties = {k: v for k, v in sub_definition.items() if k != "name"}

            sub_definitions[sub_definitions_name] = sub_definition_properties
        definition_properties = {
            k: v for k, v in definition.items() if k not in {"name", "fields", "members"}
        }

        result[definition_name] = definition_properties, sub_definitions

    return result


def _process_schema(raw_schema: str, metaschema) -> Dict:
    """Parse the schema into YAML, validate the YAML against the metaschema, and restructure."""
    unstructured_schema = yaml.safe_load(raw_schema)
    jsonschema.validate(instance=unstructured_schema, schema=metaschema)
    return _restructure(unstructured_schema)


def _changed_keys(previous_properties: Dict, current_properties: Dict) -> List[str]:
    result: List[str] = list(set(previous_properties).symmetric_difference(set(current_properties)))
    for key in set(previous_properties).intersection(set(current_properties)):
        if previous_properties[key] != current_properties[key]:
            result.append(key)
    return result


def _single_definition_changes(
    definition_name: str, old_sub_definitions: Dict, new_sub_definitions: Dict
) -> List[Change]:
    result: List[Change] = []
    for name in old_sub_definitions:
        if name in new_sub_definitions:
            for key in _changed_keys(old_sub_definitions[name], new_sub_definitions[name]):
                result.append(("sub_definition", "edit", definition_name, name, key))
        else:
            result.append(("sub_definition", "remove", definition_name, name))
    for name in new_sub_definitions:
        if name not in old_sub_definitions:
            result.append(("sub_definition", "add", definition_name, name))

    return result


def _compute_changes(old_schema: Dict, new_schema: Dict) -> List[Change]:
    result: List[Change] = []
    for name in old_schema:
        if name in new_schema:
            old_properties, old_sub_definitions = old_schema[name]
            new_properties, new_sub_definitions = new_schema[name]

            for key in _changed_keys(old_properties, new_properties):
                result.append(("definition", "edit", name, key))

            result.extend(
                _single_definition_changes(
                    name,
                    old_sub_definitions,
                    new_sub_definitions,
                )
            )
        else:
            result.append(("definition", "remove", name))
    for name in new_schema:
        if name not in old_schema:
            result.append(("definition", "add", name))

    return result


def _todo_comment(change: Change):
    action: str

    scope, operation, definition_name = change[0], change[1], change[2]
    if scope == "definition":
        match operation:
            case "add":
                action = f"addition of {definition_name}"
            case "remove":
                action = f"removal of {definition_name}"
            case "edit":
                # Mypy assumes change has length 3, the minimum possible, so this type: ignore
                # is necessary. Same below.
                property_name = change[3]  # type: ignore
                action = f"change in '{property_name}' of {definition_name}"
    else:
        sub_definition_name = change[3]  # type: ignore
        match operation:
            case "add":
                action = f"addition of {definition_name}.{sub_definition_name}"
            case "remove":
                action = f"removal of {definition_name}.{sub_definition_name}"
            case "edit":
                property_name = change[4]  # type: ignore
                action = f"change in '{property_name}' of {definition_name}.{sub_definition_name}"

    return f"# TODO: Handle {action}"


def _emit_migration_template(changes: List[Change]) -> str:
    template_file = Path(__file__).parent / "migration.py.tpl"
    template = jinja2.Template(template_file.read_text())
    return template.render(comments=[_todo_comment(change) for change in changes])


@click.command(name="generate-migrations")
@click.argument("schema")
@click.argument("upstream_branch")
@click.option("-o", "--output", help="Output the migration template here, default to stdout")
def generate_migrations(schema: str, upstream_branch: str, output: str | None) -> None:
    """Generate migrations for a schema based on its git history

    \b
    SCHEMA is the path to the schema.
    UPSTREAM_BRANCH is the commit where the previous version of the schema
    will be taken from.
    """
    metaschema = _get_metaschema()

    with open(schema) as file:
        raw_current_schema = file.read()

    raw_previous_schema = _get_raw_input_from_upstream(upstream_branch, schema)

    current_schema = _process_schema(raw_current_schema, metaschema)
    previous_schema = _process_schema(raw_previous_schema, metaschema)

    changes = _compute_changes(previous_schema, current_schema)

    migration_template = _emit_migration_template(changes)

    if output:
        with open(output, "w") as file:
            file.write(migration_template)
    else:
        print(migration_template, end="")


def setup(registry: CommandRegistry):
    registry.register(("model",), generate_migrations)

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import jinja2
import jsonschema
import yaml

from revng.internal.cli.commands_registry import Command, CommandsRegistry

Change = (
    Tuple[Literal["definition"], Literal["add", "remove"], str]
    | Tuple[Literal["definition"], Literal["edit"], str, str]
    | Tuple[Literal["sub_definition"], Literal["add", "remove"], str, str]
    | Tuple[Literal["sub_definition"], Literal["edit"], str, str, str]
)


class GenerateMigrationCommand(Command):
    def __init__(self):
        super().__init__(
            ("model", "generate-migrations"),
            "Generate migrations for a schema based on its git history",
        )

    def register_arguments(self, parser):
        parser.add_argument("schema", help="The path to the schema.")
        parser.add_argument(
            "upstream_branch",
            help="The commit where the previous version of the schema will be taken from.",
        )
        parser.add_argument(
            "-o", "--output", help="Output the migration template here, default to stdout"
        )

    def _get_metaschema(self):
        metaschema_path = Path(__file__).parent / "metaschema.yml"

        with open(metaschema_path) as file:
            metaschema = yaml.safe_load(file)

        return metaschema

    def _get_raw_input(self, schema_path: str) -> str:
        with open(schema_path) as file:
            raw_input = file.read()
        return raw_input

    def _get_raw_input_from_upstream(self, upstream_branch: str, schema_path: str) -> str:
        try:
            git_process = subprocess.run(
                ["git", "show", f"{upstream_branch}:{schema_path}"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Error while running git: {error.stderr.decode().strip()}"
            ) from error

        raw_schema = git_process.stdout.decode()

        return raw_schema

    def _restructure(self, schema: Dict) -> Dict:
        """Restructures the schema into a nested dictionaries where the `Name` field of types and
        fields/members are keys.
        """
        result: Dict[str, Any] = {}

        result = {}

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

    def _process_schema(self, raw_schema: str, metaschema) -> Dict:
        """Parse the schema into YAML, validate the YAML against the metaschema, and restructure."""
        unstructured_schema = yaml.safe_load(raw_schema)
        jsonschema.validate(instance=unstructured_schema, schema=metaschema)
        return self._restructure(unstructured_schema)

    def _changed_keys(self, previous_properties: Dict, current_properties: Dict) -> List[str]:
        result: List[str] = list(
            set(previous_properties).symmetric_difference(set(current_properties))
        )
        for key in set(previous_properties).intersection(set(current_properties)):
            if previous_properties[key] != current_properties[key]:
                result.append(key)
        return result

    def _single_definition_changes(
        self, definition_name: str, old_sub_definitions: Dict, new_sub_definitions: Dict
    ) -> List[Change]:
        result: List[Change] = []
        for name in old_sub_definitions:
            if name in new_sub_definitions:
                for key in self._changed_keys(old_sub_definitions[name], new_sub_definitions[name]):
                    result.append(("sub_definition", "edit", definition_name, name, key))
            else:
                result.append(("sub_definition", "remove", definition_name, name))
        for name in new_sub_definitions:
            if name not in old_sub_definitions:
                result.append(("sub_definition", "add", definition_name, name))

        return result

    def _compute_changes(self, old_schema: Dict, new_schema: Dict) -> List[Tuple]:
        result: List[Change] = []
        for name in old_schema:
            if name in new_schema:
                old_properties, old_sub_definitions = old_schema[name]
                new_properties, new_sub_definitions = new_schema[name]

                for key in self._changed_keys(old_properties, new_properties):
                    result.append(("definition", "edit", name, key))

                result.extend(
                    self._single_definition_changes(
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

    def _todo_comment(self, change: Change):
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
                    action = (
                        f"change in '{property_name}' of {definition_name}.{sub_definition_name}"
                    )

        return f"# TODO: Handle {action}"

    def _emit_migration_template(self, changes: List[Change]) -> str:
        template_str = """#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations.migration_base import MigrationBase


class Migration(MigrationBase):
    def __init__(self):
        pass

    def migrate(self, model):
        {%- for comment in comments %}
        {{ comment }}
        {%- endfor %}
        pass

"""
        template = jinja2.Template(template_str)
        return template.render(comments=[self._todo_comment(change) for change in changes])

    def run(self, options):
        args = options.parsed_args

        metaschema = self._get_metaschema()

        with open(args.schema) as file:
            raw_current_schema = file.read()

        raw_previous_schema = self._get_raw_input_from_upstream(args.upstream_branch, args.schema)

        current_schema = self._process_schema(raw_current_schema, metaschema)
        previous_schema = self._process_schema(raw_previous_schema, metaschema)

        changes = self._compute_changes(previous_schema, current_schema)

        migration_template = self._emit_migration_template(changes)

        if args.output:
            with open(args.output, "w") as file:
                file.write(migration_template)
        else:
            print(migration_template, end="")


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(GenerateMigrationCommand())

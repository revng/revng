from typing import Set
import idb
import yaml

from .commands_registry import Command, commands_registry, Options
from ..idb_converter import IDBConverter
from revng.model import Binary, PrimitiveType, PrimitiveTypeKind, Reference, StructType, YamlDumper


class ImportIDBCommand(Command):
    def __init__(self):
        super().__init__(
            ("model", "import", "idb"), "Extract a rev.ng model from an IDB/I64 database"
        )

    def register_arguments(self, parser):
        parser.add_argument("idb", help="Path to the IDB/I64 file to import")
        parser.add_argument(
            "--output", "-o", default="/dev/stdout", help="Output filepath (default stdout)"
        )
        parser.add_argument(
            "--output-invalid-types", help="Write list of invalid types at the given path"
        )

    def run(self, options: Options):
        idb_path = options.parsed_args.idb
        output_path = options.parsed_args.output
        output_invalid_types = options.parsed_args.output_invalid_types

        with idb.from_file(idb_path) as db:
            idb_converter = IDBConverter(db)
            revng_model = idb_converter.get_model()

        yaml_model = yaml.dump(revng_model, Dumper=YamlDumper)

        with open(output_path, "w") as output:
            output.write(yaml_model)

        if output_invalid_types:
            with open(output_invalid_types, "w") as f:
                invalid_types = get_invalid_types(revng_model)
                f.writelines(f"{line}\n" for line in invalid_types)


def get_invalid_types(model: Binary) -> Set[str]:
    invalid_types = set()
    invalid_types.update(get_unknown_types(model))
    invalid_types.update(get_invalid_structs(model))
    return invalid_types


def get_unknown_types(model: Binary):
    for t in model.Types:
        if isinstance(t, PrimitiveType):
            if t.PrimitiveKind == PrimitiveTypeKind.Invalid:
                yield Reference.create(model, t)


def get_invalid_structs(model: Binary):
    for t in model.Types:
        if not isinstance(t, StructType):
            continue

        if t.Size == 0 or len(t.Fields) == 0:
            yield Reference.create(model, t)


commands_registry.register_command(ImportIDBCommand())

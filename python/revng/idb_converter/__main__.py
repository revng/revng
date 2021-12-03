#!/usr/bin/env python3

import argparse
from typing import Set

import idb
import yaml

from .idb_converter import IDBConverter
from revng.model import Binary, Primitive, PrimitiveTypeKind, Struct, YamlDumper

argparser = argparse.ArgumentParser(description="Extracts a rev.ng model from an IDA database")
argparser.add_argument("idb", help="Path to the IDB to convert")
argparser.add_argument("--output", "-o", default="/dev/stdout", help="Output filepath (default stdout)")
argparser.add_argument("--output-invalid-types", help="Write list of invalid types at the given path")


def get_unknown_types(model: Binary):
    for type in model.Types:
        if hasattr(type, "__root__"):
            type = type.__root__

        if isinstance(type, Primitive):
            if type.PrimitiveKind == PrimitiveTypeKind.Invalid.value:
                yield make_typeref(type)


def get_invalid_structs(model: Binary):
    for type in model.Types:
        if hasattr(type, "__root__"):
            type = type.__root__

        if not isinstance(type, Struct):
            continue

        if type.Size == 0 or len(type.Fields) == 0:
            yield make_typeref(type)


def get_invalid_types(model: Binary) -> Set[str]:
    invalid_types = set()
    invalid_types.update(get_unknown_types(model))
    invalid_types.update(get_invalid_structs(model))
    return invalid_types


def _main(input_idb_path, output_file_path, output_invalid_types=None):
    mangle_names = output_invalid_types is not None

    with idb.from_file(input_idb_path) as db:
        idb_converter = IDBConverter(db, mangle_names=mangle_names)
        revng_model = idb_converter.get_model()

    yaml_model = yaml.dump(revng_model, Dumper=YamlDumper)

    with open(output_file_path, "w") as output:
        output.write(yaml_model)

    if output_invalid_types:
        with open(output_invalid_types, "w") as f:
            invalid_types = get_invalid_types(revng_model)
            f.writelines(f"{line}\n" for line in invalid_types)


def main():
    args = argparser.parse_args()
    _main(args.idb, args.output, args.output_invalid_types)


if __name__ == "__main__":
    main()

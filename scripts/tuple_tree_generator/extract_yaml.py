#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import re
import sys
from pathlib import Path

import jsonschema
import yaml

argparser = argparse.ArgumentParser()
argparser.add_argument("delimiter", help="Delimiter marking comments containing YAML to be extracted")
argparser.add_argument("headers", nargs="+")
argparser.add_argument("--output", "-o", help="Output path")

metaschema_path = Path(__file__).parent / "metaschema.yml"


def main(args):
    model_yaml_regex = re.compile(
        rf"""
        /\*\s*{args.delimiter}\s*\n    # Match start of special comment
        (?P<content>.*)          # Match everything inside in the "content" named group
        \s*\n\s*{args.delimiter}\s*\*/         # Match end comment sequence
        """,
        re.VERBOSE | re.DOTALL   # re.DOTALL enables "." to match newlines
    )

    definitions = []
    for header in args.headers:
        with open(header) as f:
            data = f.read()

        match = model_yaml_regex.search(data)
        if not match:
            sys.stderr.write(f"ERROR: could not extract model yaml in {header}")
            exit(1)
        unparsed_yaml = match.group("content")

        definition = yaml.safe_load(unparsed_yaml)
        definitions.append(definition)

    metaschema = yaml.safe_load(metaschema_path.read_text())
    jsonschema.validate(instance=definitions, schema=metaschema)

    if args.output:
        output = open(args.output, "w")
    else:
        output = sys.stdout

    yaml.safe_dump(definitions, stream=output)


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args)

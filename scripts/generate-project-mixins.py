#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# This script reads the `mixins.py.tpl` template, which contains the mixins for
# the model in `revng.project.model`. The script will render this via Jinja.

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

from revng.support.artifacts import Artifact


@dataclass
class ArtifactData:
    name: str
    type_: str


def get_container_type(data, container: str):
    container_type = data["containers"][container]
    for entry in data["container_types"]:
        if entry["class"] == container_type:
            return entry
    raise ValueError


def parse_artifacts(data) -> dict[str, list[ArtifactData]]:
    result: dict[str, list[ArtifactData]] = defaultdict(list)
    for entry in data["artifacts"]:
        if not entry["category"]["show_by_default"]:
            continue

        container_type = get_container_type(data, entry["container"])
        artifact = Artifact.make(b"", container_type["mime_type"])
        result[container_type["kind"]].append(
            ArtifactData(entry["name"], artifact.__class__.__name__)
        )

    return result


def normalize(string: str):
    return re.sub(r"[^a-zA-Z]", "_", string)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=sys.stdout)
    parser.add_argument("input")
    args = parser.parse_args()

    data = yaml.safe_load(sys.stdin)
    artifacts = parse_artifacts(data)

    input_path = Path(args.input)
    jinja_env = Environment(loader=FileSystemLoader(input_path.parent))
    jinja_env.filters["normalize"] = normalize
    template = jinja_env.get_template(input_path.name)
    rendered_template = template.render(
        binary_artifacts=artifacts["binary"],
        function_artifacts=artifacts["function"],
        typedefinition_artifacts=artifacts["type-definition"],
    )

    if args.output is sys.stdout:
        sys.stdout.write(rendered_template)
    else:
        with open(args.output, "w") as f:
            f.write(rendered_template)


if __name__ == "__main__":
    main()

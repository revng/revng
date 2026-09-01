#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys
from pathlib import Path

import click
import jinja2
import yaml

from revng.internal.cli.common import CommandRegistry

pages = [
    ("/pipeline/", "pipeline.md"),
    ("/ranks/", "ranks.md"),
    ("/kinds/", "kinds.md"),
    ("/artifacts/", "artifacts.md"),
    ("/pipes/", "pipes.md"),
    ("/analyses/", "analyses.md"),
]


@click.command(
    name="process-docs-yaml",
    help="Process YAML containing pipeline's doc",
    add_help_option=False,
)
@click.argument("template", type=str)
def process_docs_yaml(template: str) -> int:
    data = yaml.load(sys.stdin, Loader=yaml.SafeLoader)

    def link_filter(path):
        name = path.split("/")[-1]

        for prefix, page in pages:
            if path.startswith(prefix):
                path = f"{page}#{path}"
                break
        else:
            raise ValueError(f"No known prefix for {path}")

        return f"[`{name}`]({path})"

    tracker = {}
    every = 3

    def emit_edge(source, destination):
        if source not in tracker:
            tracker[source] = ("", 0)

        last, count = tracker[source]

        result = f'  "{source}" -> "{destination}":n;'

        if count % every == every - 1:
            tracker[source] = (destination, count)

        if count >= every:
            result += f'\n  "{last}" -> "{destination}" [color=transparent];'

        tracker[source] = (tracker[source][0], count + 1)

        return result

    def branch_has_successor(name: str):
        return any(b.get("from") == name for b in data["branches"].values())

    template_path = Path(template)
    loader = jinja2.FileSystemLoader(searchpath=str(template_path.parent))
    environment = jinja2.Environment(loader=loader)
    environment.filters["link"] = link_filter
    jinja_template = environment.get_template(template_path.name)

    print(
        jinja_template.render(
            data=data, emit_edge=emit_edge, branch_has_successor=branch_has_successor
        )
    )
    return 0


def setup(registry: CommandRegistry):
    registry.register((), process_docs_yaml)

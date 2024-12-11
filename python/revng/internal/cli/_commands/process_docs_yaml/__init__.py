#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys
from pathlib import Path

import jinja2
import yaml

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options

pages = [
    ("/pipeline/", "pipeline.md"),
    ("/ranks/", "ranks.md"),
    ("/kinds/", "kinds.md"),
    ("/artifacts/", "artifacts.md"),
    ("/pipes/", "pipes.md"),
    ("/analyses/", "analyses.md"),
]


class ProcessYAMLDocsCommand(Command):
    def __init__(self):
        super().__init__(("process-docs-yaml",), "Process YAML containing pipeline's doc", False)

    def register_arguments(self, parser):
        parser.add_argument("template", type=str)

    def run(self, options: Options):
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

        template_path = Path(options.parsed_args.template)
        loader = jinja2.FileSystemLoader(searchpath=str(template_path.parent))
        environment = jinja2.Environment(loader=loader)
        environment.filters["link"] = link_filter
        template = environment.get_template(template_path.name)

        print(template.render(data=data, emit_edge=emit_edge))
        return 0


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(ProcessYAMLDocsCommand())

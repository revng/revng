#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import yaml

from revng.api import make_manager, Rank
from revng.cli.commands_registry import Command, CommandsRegistry, Options
from revng.cli.support import build_command_with_loads, run

from collections import defaultdict

class DumpDocsCommand(Command):
    def __init__(self):
        # WIP: docs yaml
        super().__init__(("dump-docs",), "Dump YAML with pipeline's doc", False)

    def register_arguments(self, parser):
        pass

    def run(self, options: Options):
        manager = make_manager()

        result = {}

        # Emit pipeline description
        pipeline = {}

        # Emit container declarations
        container_declarations = []
        for container in manager.containers():
            container_declarations.append({
                "name": container.name,
                "type": "/containers/WIP"
            })

        pipeline["container_declarations"] = container_declarations

        # Emit steps
        steps = []
        pipes_set = set()
        analyses_set = set()
        analysis_steps = defaultdict(list)
        kind_artifacts = defaultdict(list)
        pipe_steps = defaultdict(list)
        for step in manager.steps():
            artifacts_single_target_filename = step.get_artifacts_single_target_filename()
            artifacts_doc = step.get_artifacts_doc()
            artifacts_kind = step.get_artifacts_kind()
            artifacts_container = step.get_artifacts_container()
            if artifacts_kind:
                kind_artifacts[artifacts_kind.name].append(f"/pipeline/steps/{step.name}")

            pipes_set = pipes_set.union(set(step.pipes()))
            analyses_set = analyses_set.union(set(step.analyses()))

            steps.append({
                "name": step.name,
                "doc": step.doc,
                "parent": f"/pipeline/steps/{step.get_parent().name}" if step.get_parent() else "",
                "analyses": [
                    f"/analyses/{analysis.name}"
                    for analysis
                    in step.analyses()
                ],
                "pipes": [
                    f"/pipes/{pipe.name}"
                    for pipe
                    in step.pipes()
                ],
                "artifact": {
                    "doc": artifacts_doc,
                    "single_target_filename": artifacts_single_target_filename,
                    "kind": f"/kinds/{artifacts_kind.name}",
                    "container": f"/containers/{artifacts_container}"
                } if artifacts_kind else {}
            })

            for pipe in step.pipes():
                pipe_steps[pipe.name].append(f"/pipeline/steps/{step.name}")

            for analysis in step.analyses():
                analysis_steps[analysis.name].append(f"/pipeline/steps/{step.name}")

        pipeline["steps"] = steps

        result["pipeline"] = pipeline

        # Emit globals
        result["globals"] = list(manager.globals_list())

        # Emit container types
        # WIP: collapse container factory

        # Emit analyses
        analyses = []
        kind_analyses = defaultdict(list)
        for analysis in analyses_set:
            analyses.append({
                "name": analysis.name,
                "doc": analysis.doc,
                "steps": analysis_steps[analysis.name],
                "arguments": [
                    {
                        "name": argument.name,
                        "acceptable_kinds": [
                            f"/kinds/{acceptable_kind.name}"
                            for acceptable_kind
                            in argument.acceptable_kinds()
                        ]
                    }
                    for argument
                    in analysis.arguments()
                ]
            })

            for argument in analysis.arguments():
                for acceptable_kind in argument.acceptable_kinds():
                    kind_analyses[acceptable_kind.name].append(f"/analyses/{analysis.name}")

        result["analyses"] = analyses

        # Emit pipes
        # WIP: container argument types
        # WIP: parse contracts or at least kinds
        pipes = []
        processed_pipes_name = set()
        for pipe in pipes_set:
            if pipe.name in processed_pipes_name:
                continue

            processed_pipes_name.add(pipe.name)
            pipes.append({
                "name": pipe.name,
                "doc": pipe.doc,
                "steps": pipe_steps[pipe.name]
            })
        result["pipes"] = pipes

        # Emit kinds
        kinds = []
        rank_kinds = defaultdict(list)
        kind_children = defaultdict(list)
        for kind in manager.kinds():
            kinds.append({
                "name": kind.name,
                "doc": kind.doc,
                "parent": f"/kinds/{kind.get_parent().name}" if kind.get_parent() else "",
                "rank": f"/ranks/{kind.rank.name}",
                "artifacts": kind_artifacts[kind.name],
                "analyses": kind_analyses[kind.name]
            })
            kind_children[kinds[-1]["parent"]].append(f"/kinds/{kind.name}")
            rank_kinds[kind.rank.name].append(f"/kinds/{kind.name}")

        for kind in kinds:
            kind["children"] = kind_children[f"/kinds/{kind['name']}"]

        result["kinds"] = kinds

        # Emit ranks
        ranks = []
        rank_children = defaultdict(list)
        for rank in Rank.ranks():
            ranks.append({
                "name": rank.name,
                "doc": rank.doc,
                "parent": f"/ranks/{rank.get_parent().name}" if rank.get_parent() else "",
                "kinds": rank_kinds[rank.name]
            })
            rank_children[ranks[-1]["parent"]].append(f"/ranks/{rank.name}")

        for rank in ranks:
            rank["children"] = rank_children[f"/ranks/{rank['name']}"]

        result["ranks"] = ranks

        print(yaml.dump(result))

        return True

import jinja2
import yaml
import sys

from pathlib import Path


def only(list):
    assert len(list) == 1
    return list[0]


def by_path(data, path):
    assert path.startswith("/")
    path = path[1:]
    for component in path.split("/"):
        if type(data) is dict:
            data = data[component]
        elif type(data) is list:
            data = only([entry for entry in data if entry["name"] == component])
        else:
            assert False

    return data

pages = [
    ("/pipeline/steps/", "pipeline.md"),
    ("/ranks/", "ranks.md"),
    ("/kinds/", "kinds.md"),
    ("/artifacts/", "artifacts.md"),
    ("/pipes/", "pipes.md"),
    ("/analyses/", "analyses.md")
]

# WIP: split
class ProcessDocsCommand(Command):
    def __init__(self):
        # WIP: docs process-yaml
        super().__init__(("docs-process-yaml",), "Process YAML containing pipeline's doc", False)

    def register_arguments(self, parser):
        parser.add_argument("template", type=str)

    def run(self, options: Options):
        data = yaml.load(sys.stdin, Loader=yaml.SafeLoader)

        def link_filter(path):
            link_text = by_path(data, path)['name']

            url = ""
            for prefix, page in pages:
                if path.startswith(prefix):
                    url = page
                    break
            
            if not url:
                raise Exception(f"Cannot create URL for {path}")

            url += "#" + path

            return f"[{link_text}]({url})"

        def name_filter(path):
            return f"{by_path(data, path)['name']}"

        template_path = Path(options.parsed_args.template)
        loader = jinja2.FileSystemLoader(searchpath=str(template_path.parent))
        environment = jinja2.Environment(loader=loader)
        environment.filters["link"] = link_filter
        environment.filters["name"] = name_filter
        template = environment.get_template(template_path.name)
                             
        print(template.render(data=data))
        return True


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(DumpDocsCommand())
    commands_registry.register_command(ProcessDocsCommand())

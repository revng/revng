#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
from functools import reduce
from pathlib import Path
from typing import Dict, List

from ariadne import ObjectType, QueryType, make_executable_schema
from graphql.type.schema import GraphQLSchema
from jinja2 import Environment, FileSystemLoader

from revng.api.analysis import Analysis
from revng.api.manager import Manager
from revng.api.rank import Rank
from revng.api.step import Step

from .static_handlers import DEFAULT_BINDABLES, analysis_mutations, run_in_executor
from .util import pascal_to_camel, str_to_snake_case


class SchemaGen:
    jenv: Environment

    def __init__(self):
        local_folder = Path(__file__).parent.resolve()
        self.jenv = Environment(loader=FileSystemLoader(str(local_folder)))
        self.jenv.filters["rank_param"] = self._rank_to_arguments
        self.jenv.filters["pascal_to_camel"] = pascal_to_camel
        self.jenv.filters["str_to_snake_case"] = str_to_snake_case
        self.jenv.filters["generate_analysis_parameters"] = self._generate_analysis_parameters

    def get_schema(self, manager: Manager) -> GraphQLSchema:
        structure = manager.pipeline_artifact_structure()
        str_schema = self._generate_schema(structure)
        bindable_gen = BindableGen(structure)
        bindables = [*DEFAULT_BINDABLES, *bindable_gen.get_bindables()]
        schema = make_executable_schema(str_schema, *bindables)  # type: ignore
        return schema

    @staticmethod
    def _rank_to_arguments(rank: Rank):
        if rank.depth == 0:
            return ""
        params = ", ".join([f"param{i+1}: String!" for i in range(rank.depth)])
        return f"({params})"

    def _generate_schema(self, structure: Dict[Rank, List[Step]]) -> str:
        steps = list(reduce(lambda x, y: [*x, *y], structure.values()))
        template = self.jenv.get_template("schema.graphql.tpl")
        render = template.render(structure=structure, steps=steps)
        return render

    def _generate_analysis_parameters(self, analysis: Analysis) -> str:
        parameters = []
        for argument in analysis.arguments():
            parameters.append(f"{str_to_snake_case(argument.name)}: String!")
        return ", ".join(parameters)


class BindableGen:
    structure: Dict[Rank, List[Step]]

    def __init__(self, structure: Dict[Rank, List[Step]]):
        self.structure = structure
        self.steps = []
        for steps in structure.values():
            self.steps.extend(steps)

    def get_bindables(self) -> List[ObjectType]:
        return [
            self.get_query_bindable(),
            *self.get_rank_bindables(),
            *self.get_analysis_bindables(),
        ]

    def get_query_bindable(self) -> QueryType:
        query_obj = QueryType()
        for rank in self.structure.keys():
            query_obj.set_field(rank.name.lower(), self.rank_handle)

        return query_obj

    def get_rank_bindables(self) -> List[ObjectType]:
        bindables = []
        for rank, steps in self.structure.items():
            rank_obj = ObjectType(rank.name.capitalize())
            bindables.append(rank_obj)
            for step in steps:
                handle = self.gen_step_handle(step)
                rank_obj.set_field(pascal_to_camel(step.name), handle)

        return bindables

    def get_analysis_bindables(self) -> List[ObjectType]:
        bindables: List[ObjectType] = []
        for step in self.steps:
            if step.analyses_count() < 1:
                continue
            analysis_mutations.set_field(pascal_to_camel(step.name), self.analysis_mutation_handle)
            step_analysis_obj = ObjectType(f"{step.name}Analyses")
            bindables.append(step_analysis_obj)
            for analysis in step.analyses():
                handle = self.gen_step_analysis_handle(step, analysis)
                step_analysis_obj.set_field(pascal_to_camel(analysis.name), handle)
        return bindables

    @staticmethod
    def rank_handle(_, info, **params):
        if not params:
            return {"_target": None}

        params_arr = []
        i = 1
        while True:
            if f"param{i}" in params:
                params_arr.append(params[f"param{i}"])
                i += 1
            else:
                return {"_target": "/".join(params_arr)}

    @staticmethod
    def analysis_mutation_handle(_, info):
        return {}

    @staticmethod
    def gen_step_handle(step: Step):
        async def rank_step_handle(obj, info, *, onlyIfReady=False):  # noqa: N803
            manager: Manager = info.context["manager"]
            return await run_in_executor(
                lambda: manager.produce_target(step.name, obj["_target"], only_if_ready=onlyIfReady)
            )

        return rank_step_handle

    @staticmethod
    def gen_step_analysis_handle(step: Step, analysis: Analysis):
        argument_mapping = {str_to_snake_case(a.name): a.name for a in analysis.arguments()}

        async def step_analysis_handle(_, info, **kwargs):
            manager: Manager = info.context["manager"]
            target_mapping = {}
            for container_name, targets in kwargs.items():
                if container_name not in argument_mapping.keys():
                    raise ValueError("Passed non-existant container name")
                target_mapping[argument_mapping[container_name]] = targets.split(",")

            result = await run_in_executor(
                lambda: manager.run_analysis(step.name, analysis.name, target_mapping)
            )
            return json.dumps(result)

        return step_analysis_handle

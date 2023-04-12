#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from ariadne import ObjectType, QueryType, make_executable_schema
from graphql.type.schema import GraphQLSchema
from jinja2 import Environment, FileSystemLoader

from revng.api.analysis import AnalysesList, Analysis
from revng.api.manager import Manager
from revng.api.rank import Rank
from revng.api.step import Step

from .event_manager import EventType, emit_event
from .static_handlers import DEFAULT_BINDABLES, analysis_mutations, invalidation_queue
from .static_handlers import run_in_executor
from .util import produce_serializer


class SchemaGenerator:
    """Class that generates a GraphQL schema object given a Manager.
    This class deals with generating the textual schema from the manager, while
    the sister class DynamicBindableGenerator is used to create the bindable objects
    that will be used to resolve the schema"""

    def __init__(self):
        local_folder = Path(__file__).parent.resolve()
        self.jinja_environment = Environment(loader=FileSystemLoader(str(local_folder)))
        filters = self.jinja_environment.filters
        filters["rank_param"] = self._rank_to_arguments
        filters["normalize"] = normalize
        filters["generate_analysis_parameters"] = self._generate_analysis_parameters

    def get_schema(self, manager: Manager) -> GraphQLSchema:
        string_schema = self._generate_schema(manager)
        bindable_generator = DynamicBindableGenerator(manager)
        bindables = [*DEFAULT_BINDABLES, *bindable_generator.get_bindables()]
        return make_executable_schema(string_schema, *bindables)  # type: ignore

    def _generate_schema(self, manager: Manager) -> str:
        template = self.jinja_environment.get_template("schema.graphql.tpl")
        rank_to_artifact_steps: Dict[Rank, List[Step]] = defaultdict(list)
        for step in manager.steps():
            step_kind = step.get_artifacts_kind()
            if step_kind is not None and step_kind.rank is not None:
                rank_to_artifact_steps[step_kind.rank].append(step)

        return template.render(
            rank_to_artifact_steps=rank_to_artifact_steps,
            steps=list(manager.steps()),
            analyses_lists=list(manager.analyses_lists()),
        )

    @staticmethod
    def _generate_analysis_parameters(analysis: Analysis) -> str:
        if len(list(analysis.arguments())) == 0:
            return ""
        return "".join(
            f"{normalize(argument.name)}: String!, " for argument in analysis.arguments()
        )

    @staticmethod
    def _rank_to_arguments(rank: Rank):
        if rank.depth == 0:
            return ""
        params = ", ".join(f"param{i+1}: String!" for i in range(rank.depth))
        return f"({params})"


class DynamicBindableGenerator:
    """Helper class that deals with generating bindables for artifacts/analyses given a manager"""

    def __init__(self, manager: Manager):
        self.manager = manager
        self.artifact_ranks: Set[Rank] = set()
        for step in self.manager.steps():
            step_kind = step.get_artifacts_kind()
            if step_kind is not None and step_kind.rank is not None:
                self.artifact_ranks.add(step_kind.rank)

    def get_bindables(self) -> List[ObjectType]:
        return [
            self.get_query_bindable(),
            self.get_analyses_list_bindables(),
            *self.get_rank_bindables(),
            *self.get_analysis_bindables(),
        ]

    def get_query_bindable(self) -> QueryType:
        query_obj = QueryType()

        for rank in self.artifact_ranks:
            query_obj.set_field(rank.name.lower(), self.rank_handle)

        return query_obj

    def get_rank_bindables(self) -> List[ObjectType]:
        bindables = []
        rank_objects = {rank: ObjectType(rank.name.capitalize()) for rank in self.artifact_ranks}

        for rank_obj in rank_objects.values():
            bindables.append(rank_obj)

        for step in self.manager.steps():
            step_kind = step.get_artifacts_kind()
            if step_kind is None or step_kind.rank is None:
                continue

            handle = self.gen_step_handle(step)
            rank_obj = rank_objects[step_kind.rank]
            rank_obj.set_field(step.name, handle)

        return bindables

    def get_analysis_bindables(self) -> List[ObjectType]:
        bindables: List[ObjectType] = []

        for step in self.manager.steps():
            if step.analyses_count() < 1:
                continue

            analysis_mutations.set_field(step.name, self.analysis_mutation_handle)
            step_analysis_obj = ObjectType(f"{step.name}Analyses")
            bindables.append(step_analysis_obj)
            for analysis in step.analyses():
                handle = self.gen_step_analysis_handle(step, analysis)
                step_analysis_obj.set_field(normalize(analysis.name), handle)

        return bindables

    def get_analyses_list_bindables(self) -> ObjectType:
        analyses_list_mutation = ObjectType("AnalysesListsMutations")
        for analyses_list in self.manager.analyses_lists():
            analyses_list_mutation.set_field(
                normalize(analyses_list.name), self.gen_analyses_list_handle(analyses_list)
            )

        return analyses_list_mutation

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
            result = await run_in_executor(
                lambda: manager.produce_target(step.name, obj["_target"], only_if_ready=onlyIfReady)
            )
            return produce_serializer(result)

        return rank_step_handle

    @staticmethod
    def gen_step_analysis_handle(step: Step, analysis: Analysis):
        argument_mapping = {normalize(a.name): a.name for a in analysis.arguments()}

        @emit_event(EventType.CONTEXT)
        async def step_analysis_handle(_, info, **kwargs):
            manager: Manager = info.context["manager"]
            target_mapping = {}

            if raw_options := kwargs.pop("options", None) is not None:
                options = json.loads(raw_options)
            else:
                options = {}

            for container_name, targets in kwargs.items():
                if container_name not in argument_mapping.keys():
                    raise ValueError("Passed non-existent container name")
                target_mapping[argument_mapping[container_name]] = targets.split(",")

            result = await run_in_executor(
                lambda: manager.run_analysis(step.name, analysis.name, target_mapping, options)
            )
            await invalidation_queue.send(str(result.invalidations))
            return json.dumps(result.result)

        return step_analysis_handle

    @staticmethod
    def gen_analyses_list_handle(analyses_list: AnalysesList):
        async def analyses_list_handle(_, info, *, options: str | None = None):
            manager: Manager = info.context["manager"]
            parsed_options = json.loads(options) if options is not None else {}

            result = await run_in_executor(manager.run_analyses_list, analyses_list, parsed_options)
            await invalidation_queue.send(str(result.invalidations))
            return json.dumps(result.result)

        return analyses_list_handle


def normalize(string: str) -> str:
    leading_digit = bool(re.match(r"\d", string))
    return re.sub(r"[^A-Za-z0-9_]", "_", f"{'_' if leading_digit else ''}{string}")

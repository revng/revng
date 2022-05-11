#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
from base64 import b64decode
from pathlib import Path
from typing import Dict, List, Optional

from starlette.datastructures import UploadFile

from ariadne import MutationType, ObjectType, QueryType, make_executable_schema, upload_scalar
from graphql.type.schema import GraphQLSchema
from jinja2 import Environment, FileSystemLoader

from revng.api.manager import Manager
from revng.api.rank import Rank
from revng.api.step import Step

from .util import clean_step_list, str_to_snake_case

query = QueryType()
mutation = MutationType()
info = ObjectType("Info")
step = ObjectType("Step")
container = ObjectType("Container")


@query.field("info")
async def resolve_root(_, info):
    return {}


@query.field("produce")
async def resolve_produce(obj, info, *, step, container, target_list, only_if_ready=False):
    manager: Manager = info.context["manager"]
    targets = target_list.split(",")
    return manager.produce_target(step, targets, container, only_if_ready)


@query.field("produce_artifacts")
async def resolve_produce_artifacts(obj, info, *, step, paths=None, only_if_ready=False):
    manager: Manager = info.context["manager"]
    target_paths = paths.split(",") if paths is not None else None
    return manager.produce_target(step, target_paths, only_if_ready=only_if_ready)


@query.field("step")
async def resolve_step(_, info, *, name):
    manager: Manager = info.context["manager"]
    step = manager.get_step(name)
    return step.as_dict() if step is not None else {}


@query.field("container")
async def resolve_container(_, info, *, name, step):
    manager: Manager = info.context["manager"]
    container_id = manager.get_container_with_name(name)
    step = manager.get_step(step)
    if step is None or container_id is None:
        return {}
    container = step.get_container(container_id)
    return container.as_dict() if container is not None else {}


@query.field("targets")
async def resolve_targets(_, info, *, pathspec):
    manager: Manager = info.context["manager"]
    targets = manager.get_all_targets()
    result = [
        {
            "name": k,
            "containers": [
                {"name": k2, "targets": [t.as_dict() for t in v2 if t.joined_path() == pathspec]}
                for k2, v2 in v.items()
            ],
        }
        for k, v in targets.items()
    ]
    clean_step_list(result)
    return result


@mutation.field("upload_b64")
async def resolve_upload_b64(_, info, *, input: str, container: str):  # noqa: A002
    manager: Manager = info.context["manager"]
    manager.set_input(container, b64decode(input))
    logging.info(f"Saved file for container {container}")
    return True


@mutation.field("upload_file")
async def resolve_upload_file(_, info, *, file: UploadFile, container: str):
    manager: Manager = info.context["manager"]
    manager.set_input(container, await file.read())
    logging.info(f"Saved file for container {container}")
    return True


@info.field("ranks")
async def resolve_ranks(_, info):
    return [x.as_dict() for x in Rank.ranks()]


@info.field("kinds")
async def resolve_root_kinds(_, info):
    manager = info.context["manager"]
    return [k.as_dict() for k in manager.kinds()]


@info.field("model")
async def resolve_root_model(_, info):
    manager = info.context["manager"]
    return manager.get_model()


@info.field("steps")
async def resolve_root_steps(_, info):
    manager: Manager = info.context["manager"]
    return [s.as_dict() for s in manager.steps()]


@step.field("containers")
async def resolve_step_containers(step_obj, info):
    if "containers" in step_obj:
        return step_obj["containers"]

    manager: Manager = info.context["manager"]
    step = manager.get_step(step_obj["name"])
    if step is None:
        return []
    containers = [step.get_container(c) for c in manager.containers()]
    return [c.as_dict() for c in containers if c is not None]


@container.field("targets")
async def resolve_container_targets(container_obj, info):
    if "targets" in container_obj:
        return container_obj["targets"]

    manager: Manager = info.context["manager"]
    targets = manager.get_targets(container_obj["_step"], container_obj["name"])
    return [t.as_dict() for t in targets]


DEFAULT_BINDABLES = (query, mutation, info, step, container, upload_scalar)


class SchemaGen:
    jenv: Environment

    def __init__(self):
        local_folder = Path(__file__).parent.resolve()
        self.jenv = Environment(loader=FileSystemLoader(str(local_folder)))
        self.jenv.filters["rank_param"] = self._rank_to_arguments
        self.jenv.filters["snake_case"] = str_to_snake_case

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
        template = self.jenv.get_template("schema.graphql.tpl")
        render = template.render(structure=structure)
        return render


class BindableGen:
    structure: Dict[Rank, List[Step]]

    def __init__(self, structure: Dict[Rank, List[Step]]):
        self.structure = structure

    def get_bindables(self) -> List[ObjectType]:
        return [self.get_query_bindable(), *self.get_rank_bindables()]

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
                rank_obj.set_field(str_to_snake_case(step.name), handle)

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
    def gen_step_handle(step: Step):
        def rank_step_handle(obj, info, *, only_if_ready=False):
            manager: Manager = info.context["manager"]
            return manager.produce_target(step.name, obj["_target"], only_if_ready=only_if_ready)

        return rank_step_handle


schema_gen = SchemaGen()


class SchemafulManager(Manager):
    _schema: Optional[GraphQLSchema]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema = None

    @property
    def schema(self) -> GraphQLSchema:
        if self._schema is None:
            self._schema = schema_gen.get_schema(self)
        return self._schema

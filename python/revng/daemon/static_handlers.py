#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import json
import logging
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, TypeVar

from starlette.datastructures import UploadFile

from ariadne import MutationType, ObjectType, QueryType, upload_scalar

from revng.api.manager import Manager
from revng.api.rank import Rank

from .util import clean_step_list

executor = ThreadPoolExecutor(1)

T = TypeVar("T")


# Python runs all coroutines in the same event loop (which is handled by a single thread)
# The scheduling is done cooperatively, so once a coroutine starts executing it will run until
# the first call to `await` or `return`.
# This can work poorly if there are long-running sync function that are executed, since those block
# the event loop. To remedy this we use a separate thread to run these functions so that the event
# loop can run other coroutines in the meantime.
# TODO: use ParamSpec and plain function when switching to python 3.10
def run_in_executor(function: Callable[[], T]) -> Awaitable[T]:
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, function)


query = QueryType()
mutation = MutationType()
info = ObjectType("Info")
step = ObjectType("Step")
container = ObjectType("Container")
analysis_mutations = ObjectType("AnalysisMutations")


@query.field("info")
async def resolve_root(_, info):
    return {}


@query.field("produce")
async def resolve_produce(obj, info, *, step, container, target_list, only_if_ready=False):
    manager: Manager = info.context["manager"]
    targets = target_list.split(",")
    return await run_in_executor(
        lambda: manager.produce_target(step, targets, container, only_if_ready)
    )


@query.field("produce_artifacts")
async def resolve_produce_artifacts(obj, info, *, step, paths=None, only_if_ready=False):
    manager: Manager = info.context["manager"]
    target_paths = paths.split(",") if paths is not None else None
    return await run_in_executor(
        lambda: manager.produce_target(step, target_paths, only_if_ready=only_if_ready)
    )


@query.field("step")
async def resolve_step(_, info, *, name):
    manager: Manager = info.context["manager"]
    step = await run_in_executor(lambda: manager.get_step(name))

    if step is None:
        return {}

    return await run_in_executor(lambda: step.as_dict())


@query.field("container")
async def resolve_container(_, info, *, name, step):
    manager: Manager = info.context["manager"]
    container_id = manager.get_container_with_name(name)
    step = await run_in_executor(lambda: manager.get_step(step))

    if step is None or container_id is None:
        return {}

    container = await run_in_executor(lambda: step.get_container(container_id))

    if container is None:
        return {}

    return await run_in_executor(lambda: container.as_dict())


@query.field("targets")
async def resolve_targets(_, info, *, pathspec):
    manager: Manager = info.context["manager"]
    targets = await run_in_executor(lambda: manager.get_all_targets())
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
    await run_in_executor(lambda: manager.set_input(container, b64decode(input)))
    logging.info(f"Saved file for container {container}")
    return True


@mutation.field("upload_file")
async def resolve_upload_file(_, info, *, file: UploadFile, container: str):
    manager: Manager = info.context["manager"]
    contents = await file.read()
    await run_in_executor(lambda: manager.set_input(container, contents))
    logging.info(f"Saved file for container {container}")
    return True


@mutation.field("run_analysis")
async def resolve_run_analysis(_, info, *, step: str, analysis: str, container: str, targets: str):
    manager: Manager = info.context["manager"]
    result = await run_in_executor(
        lambda: manager.run_analysis(step, analysis, {container: targets.split(",")})
    )
    return json.dumps(result)


@mutation.field("run_all_analyses")
async def resolve_run_all_analyses(_, info):
    manager: Manager = info.context["manager"]
    result = await run_in_executor(manager.run_all_analyses)
    return json.dumps(result)


@mutation.field("analyses")
async def mutation_analyses(_, info):
    return {}


@info.field("ranks")
async def resolve_ranks(_, info):
    return await run_in_executor(lambda: [x.as_dict() for x in Rank.ranks()])


@info.field("kinds")
async def resolve_root_kinds(_, info):
    manager: Manager = info.context["manager"]
    return await run_in_executor(lambda: [k.as_dict() for k in manager.kinds()])


@info.field("globals")
async def resolve_info_globals(_, info):
    manager: Manager = info.context["manager"]
    return await run_in_executor(lambda: list(manager.globals_list()))


@info.field("model")
async def resolve_root_model(_, info):
    manager: Manager = info.context["manager"]
    return await run_in_executor(lambda: manager.get_model())


@info.field("steps")
async def resolve_root_steps(_, info):
    manager: Manager = info.context["manager"]
    return await run_in_executor(lambda: [s.as_dict() for s in manager.steps()])


@step.field("containers")
async def resolve_step_containers(step_obj, info):
    if "containers" in step_obj:
        return step_obj["containers"]

    manager: Manager = info.context["manager"]
    step = await run_in_executor(lambda: manager.get_step(step_obj["name"]))
    if step is None:
        return []
    containers = await run_in_executor(
        lambda: [step.get_container(c) for c in manager.containers()]
    )
    return await run_in_executor(lambda: [c.as_dict() for c in containers if c is not None])


@step.field("analyses")
async def resolve_step_analyses(step_obj, info):
    manager: Manager = info.context["manager"]
    step = await run_in_executor(lambda: manager.get_step(step_obj["name"]))
    if step is None:
        return []
    return await run_in_executor(lambda: [a.as_dict() for a in step.analyses()])


@container.field("targets")
async def resolve_container_targets(container_obj, info):
    if "targets" in container_obj:
        return container_obj["targets"]

    manager: Manager = info.context["manager"]

    targets = await run_in_executor(
        lambda: manager.get_targets(container_obj["_step"], container_obj["name"])
    )
    return await run_in_executor(lambda: [t.as_dict() for t in targets])


DEFAULT_BINDABLES = (query, mutation, info, step, container, upload_scalar, analysis_mutations)

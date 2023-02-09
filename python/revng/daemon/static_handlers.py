#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import json
import logging
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Awaitable, Callable, Optional, TypeVar

from starlette.datastructures import UploadFile

from ariadne import MutationType, ObjectType, QueryType, SubscriptionType, upload_scalar

from revng.api.manager import Manager
from revng.api.rank import Rank
from revng.api.target import Target

from .event_manager import EventType, emit_event
from .multiqueue import MultiQueue
from .util import clean_step_list, produce_serializer, target_dict_to_graphql

executor = ThreadPoolExecutor(1)
invalidation_queue: MultiQueue[str] = MultiQueue()

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
subscription = SubscriptionType()
info = ObjectType("Info")
step = ObjectType("Step")
container = ObjectType("Container")
analysis_mutations = ObjectType("AnalysisMutations")


@query.field("info")
async def resolve_root(_, info):
    return {}


@query.field("produce")
async def resolve_produce(
    obj, info, *, step: str, container: str, targetList: str, onlyIfReady=False  # noqa: N803
):
    manager: Manager = info.context["manager"]
    targets = targetList.split(",")
    result = await run_in_executor(
        lambda: manager.produce_target(step, targets, container, onlyIfReady)
    )
    return produce_serializer(result)


@query.field("produceArtifacts")
async def resolve_produce_artifacts(
    obj, info, *, step: str, paths: Optional[str] = None, onlyIfReady=False  # noqa: N803
):
    manager: Manager = info.context["manager"]
    target_paths = paths.split(",") if paths is not None else None
    result = await run_in_executor(
        lambda: manager.produce_target(step, target_paths, only_if_ready=onlyIfReady)
    )
    return produce_serializer(result)


@query.field("step")
async def resolve_step(_, info, *, name):
    manager: Manager = info.context["manager"]
    step = await run_in_executor(lambda: manager.get_step(name))

    if step is None:
        return {}

    return await run_in_executor(lambda: step.as_dict())


@query.field("container")
async def resolve_container(_, info, *, name: str, step):
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
async def resolve_targets(_, info, *, pathspec: str):
    manager: Manager = info.context["manager"]
    targets = await run_in_executor(lambda: manager.get_all_targets())
    result = [
        {
            "name": k,
            "containers": [
                {
                    "name": k2,
                    "targets": [
                        target_dict_to_graphql(t.as_dict())
                        for t in v2
                        if t.joined_path() == pathspec
                    ],
                }
                for k2, v2 in v.items()
            ],
        }
        for k, v in targets.items()
    ]
    clean_step_list(result)
    return result


@query.field("target")
async def resolve_target(_, info, *, step: str, container: str, target: str) -> Optional[Target]:
    manager: Manager = info.context["manager"]
    result = await run_in_executor(
        lambda: manager.deserialize_target(f"{step}/{container}/{target}")
    )
    return result.as_dict() if result is not None else None


@mutation.field("uploadB64")
@emit_event(EventType.BEGIN)
async def resolve_upload_b64(_, info, *, input: str, container: str):  # noqa: A002
    manager: Manager = info.context["manager"]
    await run_in_executor(lambda: manager.set_input(container, b64decode(input)))
    await invalidation_queue.send("begin/input/:Binary")
    logging.info(f"Saved file for container {container}")
    return True


@mutation.field("uploadFile")
@emit_event(EventType.BEGIN)
async def resolve_upload_file(_, info, *, file: UploadFile, container: str):
    manager: Manager = info.context["manager"]
    contents = await file.read()
    await run_in_executor(lambda: manager.set_input(container, contents))
    await invalidation_queue.send("begin/input/:Binary")
    logging.info(f"Saved file for container {container}")
    return True


@mutation.field("runAnalysis")
@emit_event(EventType.CONTEXT)
async def resolve_run_analysis(_, info, *, step: str, analysis: str, container: str, targets: str):
    manager: Manager = info.context["manager"]
    result = await run_in_executor(
        lambda: manager.run_analysis(step, analysis, {container: targets.split(",")})
    )
    await invalidation_queue.send(str(result.invalidations))
    return json.dumps(result.result)


@mutation.field("runAllAnalyses")
@emit_event(EventType.CONTEXT)
async def resolve_run_all_analyses(_, info):
    manager: Manager = info.context["manager"]
    result = await run_in_executor(manager.run_all_analyses)
    await invalidation_queue.send(str(result.invalidations))
    return json.dumps(result.result)


@mutation.field("analyses")
async def mutation_analyses(_, info):
    return {}


@mutation.field("setGlobal")
@emit_event(EventType.CONTEXT)
async def mutation_set_global(_, info, *, name: str, content: str) -> bool:
    manager: Manager = info.context["manager"]
    result = await run_in_executor(lambda: manager.set_global(name, content))
    await invalidation_queue.send(str(result.invalidations))
    return result.result.unwrap()


@mutation.field("applyDiff")
@emit_event(EventType.CONTEXT)
async def mutation_apply_diff(_, info, *, globalName: str, content: str) -> bool:  # noqa: N803
    manager: Manager = info.context["manager"]
    result = await run_in_executor(lambda: manager.apply_diff(globalName, content))
    await invalidation_queue.send(str(result.invalidations))
    return result.result.unwrap()


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
    return await run_in_executor(
        lambda: [{"name": g, "content": manager.get_global(g)} for g in manager.globals_list()]
    )


@info.field("global")
async def resolve_info_global(_, info, *, name: str):
    manager: Manager = info.context["manager"]
    return await run_in_executor(lambda: manager.get_global(name))


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


@step.field("artifacts")
async def resolve_step_artifacts(step_obj, info):
    manager: Manager = info.context["manager"]
    step = await run_in_executor(lambda: manager.get_step(step_obj["name"]))
    artifacts_container = await run_in_executor(step.get_artifacts_container)
    artifacts_kind = await run_in_executor(step.get_artifacts_kind)
    artifacts_single_target_filename = await run_in_executor(
        step.get_artifacts_single_target_filename
    )
    if (
        artifacts_container is None
        or artifacts_kind is None
        or artifacts_single_target_filename is None
    ):
        return None

    return {
        "kind": artifacts_kind.as_dict(),
        "container": artifacts_container.as_dict(),
        "singleTargetFilename": artifacts_single_target_filename,
    }


@container.field("targets")
async def resolve_container_targets(container_obj, info):
    if "targets" in container_obj:
        return container_obj["targets"]

    manager: Manager = info.context["manager"]

    targets = await run_in_executor(
        lambda: manager.get_targets(container_obj["_step"], container_obj["name"])
    )
    return await run_in_executor(lambda: [target_dict_to_graphql(t.as_dict()) for t in targets])


@info.field("verifyGlobal")
async def info_verify_global(_, info, *, name: str, content: str) -> bool:
    manager: Manager = info.context["manager"]
    result = await run_in_executor(lambda: manager.verify_global(name, content))
    return result.unwrap()


@info.field("verifyDiff")
async def info_verify_diff(_, info, *, globalName: str, content: str) -> bool:  # noqa: N803
    manager: Manager = info.context["manager"]
    result = await run_in_executor(lambda: manager.verify_diff(globalName, content))
    return result.unwrap()


@subscription.source("invalidations")
async def invalidations_generator(_, info) -> AsyncGenerator[str, None]:
    with invalidation_queue.stream() as stream:
        async for message in stream:
            yield message


@subscription.field("invalidations")
async def invalidations(message: str, info):
    return message


DEFAULT_BINDABLES = (
    query,
    mutation,
    subscription,
    info,
    step,
    container,
    upload_scalar,
    analysis_mutations,
)

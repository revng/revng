#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import json
import sys
from graphlib import TopologicalSorter
from typing import Awaitable, Callable, Iterable, List, Tuple

import aiohttp
import yaml
from aiohttp import ClientSession, ClientTimeout
from gql import Client, gql
from gql.client import AsyncClientSession
from gql.transport.aiohttp import AIOHTTPTransport

from revng.pipeline_description import Artifacts, YamlLoader  # type: ignore

from .daemon_handler import DaemonHandler

Runner = Callable[[AsyncClientSession], Awaitable[None]]


async def run_on_daemon(handler: DaemonHandler, runners: Iterable[Runner]):
    await handler.wait_for_start()
    await check_server_up(handler.url)

    connector, address = get_connection(handler.url)
    transport = AIOHTTPTransport(
        f"http://{address}/graphql/",
        client_session_args={"connector": connector, "timeout": ClientTimeout()},
    )
    async with Client(
        transport=transport, fetch_schema_from_transport=True, execute_timeout=None
    ) as client:
        for runner in runners:
            await runner(client)


def upload_file(executable_path: str):
    async def runner(client: AsyncClientSession):
        upload_q = gql(
            """
            mutation upload($file: Upload!) {
                uploadFile(file: $file, container: "input")
            }"""
        )

        with open(executable_path, "rb") as binary_file:
            await client.execute(upload_q, variable_values={"file": binary_file}, upload_files=True)
        log("Upload complete")

    return runner


def run_analyses_lists(analyses_lists: List[str]):
    async def runner(client: AsyncClientSession):
        q = gql("""{ pipelineDescription }""")
        description_req = await client.execute(q)
        description = yaml.load(description_req["pipelineDescription"], Loader=YamlLoader)

        list_names = [al.Name for al in description.AnalysesLists]

        for list_name in analyses_lists:
            assert list_name in list_names, f"Missing analyses list {list_name}"

            log(f"Running analyses list {list_name}")
            index = (await client.execute(gql("{ index: contextCommitIndex }")))["index"]
            q = gql(
                "mutation {"
                + f'runAnalysesList(name: "{list_name}", index: "{index}")'
                + "{ __typename ... on SimpleError { errorType message } } }"
            )
            res = await client.execute(q)

            if res["runAnalysesList"]["__typename"] != "Diff":
                log(f"Analysis failed: {json.dumps(res, indent=2)}")
                assert False

    return runner


def produce_artifacts(filter_: List[str] | None = None):
    async def runner(client: AsyncClientSession):
        q = gql("""{ pipelineDescription contextCommitIndex }""")
        req = await client.execute(q)
        description = yaml.load(req["pipelineDescription"], Loader=YamlLoader)
        index = req["contextCommitIndex"]

        if filter_ is None:
            filtered_steps = list(description.Steps)
        else:
            filtered_steps = [
                step
                for step in description.Steps
                if step.Component in filter_ or step.Name == "begin"
            ]

        steps = {step.Name: step for step in filtered_steps}
        topo_sorter: TopologicalSorter = TopologicalSorter()
        for step in steps.values():
            if step.Parent != "":
                if step.Parent in steps:
                    topo_sorter.add(step.Name, step.Parent)
                else:
                    topo_sorter.add(step.Name, "begin")

        for step_name in topo_sorter.static_order():
            step = steps[step_name]
            if step.Artifacts == Artifacts():
                continue

            artifacts_container = step.Artifacts.Container
            artifacts_kind = step.Artifacts.Kind

            q = gql(
                """
            query cq($step: String!, $container: String!) {
                targets(step: $step, container: $container) {
                    serialized
                }
            }"""
            )
            arguments = {"step": step_name, "container": artifacts_container}
            res = await client.execute(q, arguments)

            target_list = {
                target["serialized"]
                for target in res["targets"]
                if target["serialized"].endswith(f":{artifacts_kind}")
            }
            targets = ",".join(target_list)

            log(f"Producing {step_name}/{artifacts_container}/*:{artifacts_kind}")
            q = gql(
                """
            query($step: String!, $container: String!, $target: String!, $index: BigInt!) {
                produce(step: $step, container: $container, targetList: $target, index: $index) {
                    __typename
                    ... on Produced {
                        result
                    }
                }
            }"""
            )
            result = await client.execute(q, {**arguments, "target": targets, "index": index})
            assert result["produce"]["__typename"] == "Produced"
            json_result = json.loads(result["produce"]["result"])
            assert target_list == set(json_result.keys()), "Some targets were not produced"

    return runner


async def check_server_up(url: str):
    connector, address = get_connection(url)
    session = ClientSession(connector=connector, timeout=ClientTimeout())
    for _ in range(10):
        try:
            async with session.get(f"http://{address}/status") as req:
                if req.status == 200:
                    connector.close()
                    return
                await asyncio.sleep(1.0)
        except aiohttp.ClientConnectionError:
            await asyncio.sleep(1.0)
    connector.close()
    raise ValueError()


def get_connection(url) -> Tuple[aiohttp.BaseConnector, str]:
    if url.startswith("unix:"):
        return (aiohttp.UnixConnector(url.replace("unix:", "", 1)), "dummy")
    return (aiohttp.TCPConnector(), url)


def log(string: str):
    sys.stderr.write(f"{string}\n")
    sys.stderr.flush()

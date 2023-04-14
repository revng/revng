#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import json
import sys
from graphlib import TopologicalSorter
from typing import Tuple

import aiohttp
from aiohttp import ClientSession, ClientTimeout
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from .daemon_handler import DaemonHandler


async def run_self_test(handler: DaemonHandler, executable_path: str, has_revng_c: bool):
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
        upload_q = gql(
            """
            mutation upload($file: Upload!) {
                uploadFile(file: $file, container: "input")
            }"""
        )

        with open(executable_path, "rb") as binary_file:
            await client.execute(upload_q, variable_values={"file": binary_file}, upload_files=True)
        log("Upload complete")

        q = gql("""{ info { analysesLists { name }}}""")
        analyses_lists = await client.execute(q)
        list_names = [al["name"] for al in analyses_lists["info"]["analysesLists"]]

        assert "revng-initial-auto-analysis" in list_names, "Missing revng initial auto-analysis"

        await client.execute(
            gql("""mutation { runAnalysesList(name: "revng-initial-auto-analysis") }""")
        )

        if has_revng_c:
            assert (
                "revng-c-initial-auto-analysis" in list_names
            ), "Missing revng-c initial auto-analysis"

        if "revng-c-initial-auto-analysis" in list_names:
            await client.execute(
                gql("""mutation { runAnalysesList(name: "revng-c-initial-auto-analysis") }""")
            )

        log("Autoanalysis complete")

        q = gql(
            """{ info { steps {
                name
                parent
                artifacts {
                    kind { name }
                    container { name }
                }
            }}}"""
        )

        result = await client.execute(q)

        steps = {step["name"]: step for step in result["info"]["steps"]}
        topo_sorter: TopologicalSorter = TopologicalSorter()
        for step in steps.values():
            if step["parent"] is not None:
                topo_sorter.add(step["name"], step["parent"])

        for step_name in topo_sorter.static_order():
            step = steps[step_name]
            if step["artifacts"] is None:
                continue

            artifacts_container = step["artifacts"]["container"]["name"]
            artifacts_kind = step["artifacts"]["kind"]["name"]

            q = gql(
                """
            query cq($step: String!, $container: String!) {
                container(name: $container, step: $step) {
                    targets { serialized }
                }
            }"""
            )
            arguments = {"step": step_name, "container": artifacts_container}
            res = await client.execute(q, arguments)

            target_list = {
                target["serialized"]
                for target in res["container"]["targets"]
                if target["serialized"].endswith(f":{artifacts_kind}")
            }
            targets = ",".join(target_list)

            log(f"Producing {step_name}/{artifacts_container}/*:{artifacts_kind}")
            q = gql(
                """
            query($step: String!, $container: String!, $target: String!) {
                produce(step: $step, container: $container, targetList: $target)
            }"""
            )
            result = await client.execute(q, {**arguments, "target": targets})
            json_result = json.loads(result["produce"])
            assert target_list == set(json_result.keys()), "Some targets were not produced"


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

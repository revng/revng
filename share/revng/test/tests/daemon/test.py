#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import io
import json
import os
import signal
import sys
from subprocess import STDOUT, Popen, TimeoutExpired
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Any, AsyncGenerator

import yaml
from aiohttp.client import ClientConnectionError, ClientSession, ClientTimeout
from aiohttp.connector import UnixConnector
from aiohttp.tracing import TraceConfig, TraceRequestChunkSentParams, TraceRequestEndParams
from aiohttp.tracing import TraceRequestHeadersSentParams
from gql import Client, gql
from gql.client import AsyncClientSession
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.exceptions import TransportQueryError
from pytest import Config, ExceptionInfo, TestReport, mark
from pytest_asyncio import fixture

from revng.pipeline_description import YamlLoader  # type: ignore

pytestmark = mark.asyncio

FILTER_ENV = [
    "STARLETTE_DEBUG",
    "REVNG_NOTIFY_FIFOS",
    "REVNG_ORIGINS",
    "REVNG_DATA_DIR",
    "REVNG_PROJECT_ID",
]


def log(string: Any):
    sys.stderr.write(f"{string}\n")
    sys.stderr.flush()


def print_fd(fd: int):
    os.lseek(fd, 0, io.SEEK_SET)
    out_read = os.fdopen(fd, "r")
    log(out_read.read())


async def check_server_up(connector):
    async with ClientSession(
        connector=connector, connector_owner=False, timeout=ClientTimeout(total=2.0)
    ) as session:
        for _ in range(10):
            try:
                async with session.get("http://dummyhost/status") as req:
                    if req.status == 200:
                        return
                    await asyncio.sleep(1.0)
            except ClientConnectionError:
                await asyncio.sleep(1.0)
        raise ValueError()


async def header_trace(session, trace_config_ctx, params: TraceRequestHeadersSentParams):
    log(f"URL: {params.url}")
    log(f"METHOD: {params.method}")
    log(f"HEADERS: {params.headers}")


async def payload_trace(session, trace_config_ctx, params: TraceRequestChunkSentParams):
    log(f"DATA: {params.chunk[:256]!r}")


async def response_trace(session, trace_config_ctx, params: TraceRequestEndParams):
    log(f"RESPONSE: {params.response}")


@fixture
async def client(pytestconfig: Config, request) -> AsyncGenerator[AsyncClientSession, None]:
    temp_dir = TemporaryDirectory(prefix="revng-daemon-test-")
    log_file = TemporaryFile("wb+", prefix="revng-daemon-test-log-")
    socket_path = f"{temp_dir.name}/daemon.sock"
    new_env = {k: v for k, v in os.environ.items() if k not in FILTER_ENV}
    process = Popen(
        [
            "revng",
            "daemon",
            "-b",
            f"unix:{socket_path}",
        ],
        stdout=log_file.fileno(),
        stderr=STDOUT,
        env=new_env,
    )

    def stop_daemon():
        if process.returncode is not None:
            return process.returncode

        process.send_signal(signal.SIGINT)
        try:
            return process.wait(30.0)
        except TimeoutExpired:
            process.send_signal(signal.SIGKILL)

        return process.wait()

    def error_handler(e: BaseException):
        return_code = stop_daemon()
        log_file.seek(0)

        log("\n\n########## BEGIN DAEMON LOG ##########\n\n")
        log(log_file.read().decode("utf-8"))
        log("\n\n########## END DAEMON LOG ##########\n\n")
        log(f"The daemon exited with code {return_code}\n")

        raise e

    connector = UnixConnector(socket_path, force_close=True)

    try:
        await check_server_up(connector)
    except ValueError as e:
        error_handler(e)

    binary = pytestconfig.getoption("binary")
    tracing = TraceConfig()
    tracing.on_request_headers_sent.append(header_trace)
    tracing.on_request_chunk_sent.append(payload_trace)
    tracing.on_request_end.append(response_trace)
    transport = AIOHTTPTransport(
        "http://dummyhost/graphql/",
        client_session_args={
            "connector": connector,
            "timeout": ClientTimeout(),
            "trace_configs": [tracing],
        },
    )
    gql_client = Client(transport=transport, fetch_schema_from_transport=True, execute_timeout=None)

    upload_q = gql(
        """
        mutation upload($file: Upload!) {
            uploadFile(file: $file, container: "input")
        }
    """
    )

    try:
        async with gql_client as session:
            with open(binary, "rb") as binary_file:
                await session.execute(
                    upload_q, variable_values={"file": binary_file}, upload_files=True
                )
            yield session
    except Exception as e:
        error_handler(e)

    test_report: TestReport = request.node.rep_call
    if test_report.failed:
        if isinstance(test_report.longrepr, ExceptionInfo):
            error_handler(test_report.longrepr.value)
        else:
            error_handler(ValueError(test_report.longreprtext))

    # Terminate the daemon gracefully
    return_code = stop_daemon()

    # Check that the daemon exited cleanly
    if return_code != 0:
        error_handler(ValueError(f"Daemon exited with non-zero return code: {return_code}"))


async def test_pipeline_description(client):
    q = gql("{ pipelineDescription }")
    result = await client.execute(q)
    yaml.load(result["pipelineDescription"], Loader=YamlLoader)


async def get_description(client):
    q = gql("{ pipelineDescription }")
    result = await client.execute(q)
    return yaml.load(result["pipelineDescription"], Loader=YamlLoader)


async def test_info(client):
    desc = await get_description(client)

    binary_kind = next(k for k in desc.Kinds if k.Name == "binary")
    isolated_kind = next(k for k in desc.Kinds if k.Name == "isolated-root")
    assert binary_kind.Rank == "binary"
    assert binary_kind.Parent == ""
    assert isolated_kind.Rank == "binary"
    assert isolated_kind.Parent == "root"

    root_rank = next(r for r in desc.Ranks if r.Name == "binary")
    function_rank = next(r for r in desc.Ranks if r.Name == "function")
    assert root_rank.Depth == 0
    assert root_rank.Parent == ""
    assert function_rank.Depth == 1
    assert function_rank.Parent == "binary"

    begin_step = next(s for s in desc.Steps if s.Name == "begin")
    initial_step = next(s for s in desc.Steps if s.Name == "initial")
    assert begin_step.Parent == ""
    assert initial_step.Parent == "begin"

    container_names = [c.Name for c in desc.Containers]
    assert "root.bc.zstd" in container_names
    assert "functions.bc.zstd" in container_names
    assert "input" in container_names

    input_container = next(c for c in desc.Containers if c.Name == "input")
    assert input_container.MIMEType != ""

    auto_analysis_found = False
    for alist in desc.AnalysesLists:
        if alist.Name == "revng-initial-auto-analysis":
            auto_analysis_found = True
        assert len(alist.Analyses) > 0, f"Analyses list {alist.Name} has 0 analyses"
    assert auto_analysis_found, "revng-initial-auto-analysis not found in analyses lists"


async def test_info_global(client):
    desc = await get_description(client)
    assert "model.yml" in desc.Globals

    result = await client.execute(gql('{ getGlobal(name: "model.yml") }'))
    assert result["getGlobal"] is not None


async def get_index(client):
    req = await client.execute(gql("{ index: contextCommitIndex }"))
    return req["index"]


async def run_preliminary_analyses(client):
    index = await get_index(client)
    await client.execute(
        gql(
            """mutation($ctt: String!, $index: BigInt!) {
                runAnalysis(step: "initial", analysis: "import-binary",
                            containerToTargets: $ctt, index: $index) {
                    __typename
                }
            }"""
        ),
        {"ctt": json.dumps({"input": [":binary"]}), "index": index},
    )


async def test_lift(client):
    await run_preliminary_analyses(client)
    index = await get_index(client)
    result = await client.execute(
        gql(f'{{ produceArtifacts(step: "lift", paths: "", index: "{index}") {{ __typename }} }}')
    )
    assert result["produceArtifacts"]["__typename"] == "Produced"


async def test_lift_ready_fail(client):
    await run_preliminary_analyses(client)
    index = await get_index(client)
    q = gql(
        f'{{ produceArtifacts(step: "lift", paths: ":binary", onlyIfReady: true, index: "{index}")'
        + "{ __typename } }"
    )

    try:
        await client.execute(q)
        raise ValueError("Exception expected")
    except TransportQueryError as e:
        assert len(e.errors) == 1
        assert e.errors[0]["message"] == "Path components need to equal kind rank"


async def test_get_model(client):
    await run_preliminary_analyses(client)
    index = await get_index(client)
    await client.execute(
        gql(f'{{ produceArtifacts(step: "lift", paths: "", index: "{index}") {{ __typename }} }}')
    )

    result = await client.execute(gql('{ getGlobal(name: "model.yml") }'))
    assert result["getGlobal"] is not None


async def test_targets(client):
    q = gql(
        """
    {
        begin: targets(step: "begin", container: "input") {
            kind
            ready
        }

        lift: targets(step: "lift", container: "root.bc.zstd") {
            kind
            ready
        }
    }
    """
    )
    result = await client.execute(q)

    binary_target = next(t for t in result["begin"] if t["kind"] == "binary")
    lift_target = next(t for t in result["lift"] if t["kind"] == "root")
    assert binary_target["ready"]
    assert not lift_target["ready"]


async def test_produce(client):
    await run_preliminary_analyses(client)
    index = await get_index(client)
    q = gql(
        '{ produce(step: "lift", container: "root.bc.zstd", targetList: ":root", '
        + f'index: "{index}")'
        + "{ __typename } }"
    )
    result = await client.execute(q)

    assert result["produce"]["__typename"] == "Produced"


async def test_produce_artifact(client):
    await run_preliminary_analyses(client)
    index = await get_index(client)
    q = gql(f'{{ produceArtifacts(step: "lift", index: "{index}") {{ __typename }} }}')
    result = await client.execute(q)

    assert "produceArtifacts" in result
    assert result["produceArtifacts"]["__typename"] == "Produced"


async def test_function_endpoint(client):
    await run_preliminary_analyses(client)
    index = await get_index(client)
    q = gql(
        """mutation($ctt: String!, $index: BigInt!) {
                runAnalysis(step: "lift", analysis: "detect-abi",
                            containerToTargets: $ctt, index: $index) {
                    __typename
                }
        }"""
    )
    await client.execute(q, {"ctt": json.dumps({"root.bc.zstd": [":root"]}), "index": index})

    q = gql(
        """{
            targets(step: "isolate", container: "functions.bc.zstd") {
                serialized
            }
        }"""
    )
    result = await client.execute(q)

    first_function = next(t for t in result["targets"] if not t["serialized"].startswith(":"))
    index = await get_index(client)
    q = gql(
        """query function($param1: String!, $index: BigInt!) {
            produceArtifacts(step: "isolate", paths: $param1, index: $index) { __typename }
        }"""
    )
    result = await client.execute(
        q, {"param1": first_function["serialized"].rsplit(":", 1)[0], "index": index}
    )

    assert result["produceArtifacts"]["__typename"] == "Produced"


async def test_analysis_kind_check(client):
    ctt = json.dumps({"root.bc.zstd": [":isolated-root"]})
    index = await get_index(client)
    q = gql(
        """mutation($ctt: String!, $index: BigInt!) {
            runAnalysis(step: "lift", analysis: "detect-abi",
                        containerToTargets: $ctt, index: $index) {
                __typename
            }
        }"""
    )
    try:
        await client.execute(q, {"ctt": ctt, "index": index})
        raise ValueError("Expected exception")
    except TransportQueryError as e:
        assert len(e.errors) == 1
        assert "Wrong kind for analysis" in e.errors[0]["message"]


async def test_analyses_list(client):
    index = await get_index(client)
    q = gql(
        "mutation { "
        + f'runAnalysesList(name: "revng-initial-auto-analysis", index: "{index}")'
        + "{ __typename } }"
    )
    result = await client.execute(q)

    assert result["runAnalysesList"]["__typename"] == "Diff"

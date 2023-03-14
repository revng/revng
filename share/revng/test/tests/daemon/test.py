#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import io
import os
import sys
from signal import SIGINT
from subprocess import PIPE, STDOUT, Popen
from tempfile import TemporaryDirectory
from typing import Any, AsyncGenerator

from aiohttp.client import ClientConnectionError, ClientSession, ClientTimeout
from aiohttp.connector import UnixConnector
from aiohttp.tracing import TraceConfig, TraceRequestChunkSentParams, TraceRequestEndParams
from aiohttp.tracing import TraceRequestHeadersSentParams
from gql import Client, gql
from gql.client import AsyncClientSession
from gql.transport.aiohttp import AIOHTTPTransport
from pytest import Config, ExceptionInfo, TestReport, mark
from pytest_asyncio import fixture

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
    temp_dir = TemporaryDirectory()
    socket_path = f"{temp_dir.name}/daemon.sock"
    new_env = {k: v for k, v in os.environ.items() if k not in FILTER_ENV}
    process = Popen(
        [
            "revng",
            "daemon",
            "-b",
            f"unix:{socket_path}",
        ],
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        env=new_env,
    )

    def error_handler(e: BaseException):
        # If the daemon hasn't stopped, do so gracefully
        if process.poll() is None:
            process.terminate()
        log("\n\n########## BEGIN DAEMON LOG ##########\n\n")
        log(process.communicate()[0])
        log("\n\n########## END DAEMON LOG ##########\n\n")
        log(f"The daemon exited with code {process.returncode}\n")
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
    process.send_signal(SIGINT)
    return_code = process.wait()

    # Check that the daemon exited cleanly
    if return_code != 0:
        error_handler(ValueError(f"Daemon exited with non-zero return code: {return_code}"))


async def test_info(client):
    q = gql(
        """
    {
        info {
            kinds {
                name
                rank
                parent
            }
            ranks {
                name
                depth
                parent
            }
            steps {
                name
            }
        }
    }
    """
    )

    result = await client.execute(q)

    binary_kind = next(k for k in result["info"]["kinds"] if k["name"] == "Binary")
    isolated_kind = next(k for k in result["info"]["kinds"] if k["name"] == "IsolatedRoot")
    assert binary_kind["rank"] == "binary"
    assert binary_kind["parent"] is None
    assert isolated_kind["rank"] == "binary"
    assert isolated_kind["parent"] == "Root"

    root_rank = next(r for r in result["info"]["ranks"] if r["name"] == "binary")
    function_rank = next(r for r in result["info"]["ranks"] if r["name"] == "function")
    assert root_rank["depth"] == 0
    assert root_rank["parent"] is None
    assert function_rank["depth"] == 1
    assert function_rank["parent"] == "binary"

    step_names = [s["name"] for s in result["info"]["steps"]]
    assert "begin" in step_names


async def test_info_global(client):
    q = gql(
        """
    {
        info {
            globals {
                name
                content
            }
            model: global(name: "model.yml")
        }
    }
    """
    )

    result = await client.execute(q)

    model = next(r for r in result["info"]["globals"] if r["name"] == "model.yml")
    model_content = result["info"]["model"]

    assert model is not None
    assert model_content is not None
    assert model["content"] == model_content


async def run_preliminary_analyses(client):
    await client.execute(
        gql(
            """
    mutation {
        analyses {
            Import {
                ImportBinary(input: ":Binary"),
                AddPrimitiveTypes(input: ":Binary")
            }
        }
    }"""
        )
    )


async def test_lift(client):
    await run_preliminary_analyses(client)

    result = await client.execute(gql("{ binary { Lift } }"))
    assert result["binary"]["Lift"] is not None


@mark.xfail(raises=Exception)
async def test_lift_ready_fail(client):
    await run_preliminary_analyses(client)
    await client.execute(gql("{ binary { Lift(onlyIfReady: true) } }"))


@mark.xfail(raises=Exception)
async def test_invalid_step(client):
    q = gql('{ step(name: "this_step_does_not_exist") { name } }')
    await client.execute(q)


async def test_valid_steps(client):
    q = gql(
        """
    {
        begin: step(name: "begin") {
            name
            parent
        }
        import: step(name: "Import") {
            name
            parent
        }
    }
    """
    )
    result = await client.execute(q)

    assert result["begin"]["name"] == "begin"
    assert result["begin"]["parent"] is None

    assert result["import"]["name"] == "Import"
    assert result["import"]["parent"] == "begin"


async def test_begin_has_containers(client):
    q = gql(
        """
    {
        step(name: "begin") {
            containers {
                name
                mime
            }
        }
        container(name: "input", step: "begin") {
            name
            mime
        }
    }
    """
    )
    result = await client.execute(q)

    container_names = [c["name"] for c in result["step"]["containers"]]
    input_container = next(c for c in result["step"]["containers"] if c["name"] == "input")
    assert "module.ll" in container_names
    assert "input" in container_names
    assert input_container["mime"] is not None
    assert input_container["mime"] == result["container"]["mime"]


async def test_get_model(client):
    await run_preliminary_analyses(client)
    await client.execute(gql("{ binary { Lift } }"))

    result = await client.execute(gql("{ info { model } }"))
    assert result["info"]["model"] is not None


async def test_targets_from_step(client):
    q = gql(
        """
    {
        begin: step(name: "begin") {
            containers {
                name
                targets {
                    serialized
                    kind
                    ready
                }
            }
        }

        lift: step(name: "Lift") {
            containers {
                name
                targets {
                    kind
                    ready
                }
            }
        }
    }
    """
    )
    result = await client.execute(q)

    binary_target = next(
        t
        for t in next(c for c in result["begin"]["containers"] if c["name"] == "input")["targets"]
        if t["kind"] == "Binary"
    )
    lift_target = next(
        t
        for t in next(c for c in result["lift"]["containers"] if c["name"] == "module.ll")[
            "targets"
        ]
        if t["kind"] == "Root"
    )
    assert binary_target["ready"]
    assert not lift_target["ready"]


async def test_targets(client):
    q = gql(
        """
    {
        targets(pathspec: "") {
            name
        }
    }
    """
    )
    result = await client.execute(q)

    names = [s["name"] for s in result["targets"]]
    assert "begin" in names


async def test_produce(client):
    await run_preliminary_analyses(client)
    q = gql('{ produce(step: "Lift", container: "module.ll", targetList: ":Root") }')
    result = await client.execute(q)

    assert "produce" in result, result["produce"] != ""


async def test_produce_artifact(client):
    await run_preliminary_analyses(client)
    q = gql('{ produceArtifacts(step: "Lift") }')
    result = await client.execute(q)

    assert "produceArtifacts" in result, result["produceArtifacts"] != ""


async def test_function_endpoint(client):
    await run_preliminary_analyses(client)
    await client.execute(
        gql(
            """
    mutation {
        analyses {
            Lift {
                DetectABI(module_ll: ":Root")
            }
        }
    }
    """
        )
    )

    q = gql(
        """
    {
        container(name: "module.ll", step: "Isolate") {
            targets {
                serialized
            }
        }
    }
    """
    )
    result = await client.execute(q)

    first_function = next(
        t for t in result["container"]["targets"] if not t["serialized"].startswith(":")
    )["serialized"].rsplit(":", 1)[0]
    q = gql(
        """
    query function($param1: String!) {
        function(param1: $param1) {
            Isolate
        }
    }
    """
    )
    result = await client.execute(q, {"param1": first_function})

    assert result["function"]["Isolate"] is not None


@mark.xfail(raises=Exception)
async def test_analysis_kind_check(client):
    await client.execute(
        gql(
            """
    mutation {
        analyses {
            Lift {
                DetectABI(module_ll: ":IsolatedRoot")
            }
        }
    }
    """
        )
    )

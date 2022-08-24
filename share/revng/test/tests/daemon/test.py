#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import io
import os
from subprocess import Popen
from time import sleep
from typing import Generator
from urllib.error import URLError
from urllib.request import urlopen

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from psutil import Process
from pytest import Config, fixture, mark

FILTER_ENV = [
    "STARLETTE_DEBUG",
    "REVNG_NOTIFY_FIFOS",
    "REVNG_ORIGINS",
    "REVNG_DATA_DIR",
    "REVNG_PROJECT_ID",
]


def print_fd(fd: int):
    os.lseek(fd, 0, io.SEEK_SET)
    out_read = os.fdopen(fd, "r")
    print(out_read.read())


def get_listen_port(pid: int) -> int:
    psutil_process = Process(pid)
    while True:
        connections = psutil_process.connections()
        for connection in connections:
            if connection.raddr == () and connection.status == "LISTEN":
                return connection.laddr.port
        sleep(0.5)


def check_server_up(port: int):
    for _ in range(10):
        try:
            req = urlopen(f"http://127.0.0.1:{port}/status", timeout=1.0)
            if req.code == 200:
                return
            sleep(1.0)
        except (URLError, TimeoutError):
            sleep(1.0)
    raise ValueError()


@fixture
def client(pytestconfig: Config, request) -> Generator[Client, None, None]:
    out_fd = os.memfd_create("flask_debug", 0)
    out = os.fdopen(out_fd, "w")

    new_env = {k: v for k, v in os.environ.items() if k not in FILTER_ENV}
    process = Popen(["revng", "daemon", "-p", "0"], stdout=out, stderr=out, text=True, env=new_env)
    port = get_listen_port(process.pid)

    try:
        check_server_up(port)
    except ValueError as e:
        print_fd(out_fd)
        raise e

    binary = pytestconfig.getoption("binary")
    transport = RequestsHTTPTransport(f"http://127.0.0.1:{port}/graphql/")
    gql_client = Client(transport=transport, fetch_schema_from_transport=True)

    upload_q = gql(
        """
        mutation upload($file: Upload!) {
            uploadFile(file: $file, container: "input")
        }
    """
    )
    with open(binary, "rb") as binary_file:
        gql_client.execute(upload_q, variable_values={"file": binary_file}, upload_files=True)
    yield gql_client

    process.terminate()
    process.wait()

    if request.node.rep_call.failed:
        print_fd(out_fd)


def test_info(client):
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

    result = client.execute(q)

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


def test_info_global(client):
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

    result = client.execute(q)

    model = next(r for r in result["info"]["globals"] if r["name"] == "model.yml")
    model_content = result["info"]["model"]

    assert model is not None
    assert model_content is not None
    assert model["content"] == model_content


def run_preliminary_analyses(client):
    client.execute(
        gql(
            """
    mutation {
        analyses {
            Import {
                ImportBinary(input: ":Binary"),
                AddPrimitiveTypes(input: ":Binary")
            }
        }
    }
    """
        )
    )


def test_lift(client):
    run_preliminary_analyses(client)

    result = client.execute(gql("{ binary { Lift } }"))
    assert result["binary"]["Lift"] is not None


@mark.xfail(raises=Exception)
def test_lift_ready_fail(client):
    client.execute(gql("{ binary { Lift(onlyIfReady: true) } }"))


@mark.xfail(raises=Exception)
def test_invalid_step(client):
    q = gql(
        """
    {
        step(name: "this_step_does_not_exist") {
            name
        }
    }
    """
    )
    client.execute(q)


def test_valid_steps(client):
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
    result = client.execute(q)

    assert result["begin"]["name"] == "begin"
    assert result["begin"]["parent"] is None

    assert result["import"]["name"] == "Import"
    assert result["import"]["parent"] == "begin"


def test_begin_has_containers(client):
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
    result = client.execute(q)

    container_names = [c["name"] for c in result["step"]["containers"]]
    input_container = next(c for c in result["step"]["containers"] if c["name"] == "input")
    assert "module.ll" in container_names
    assert "input" in container_names
    assert input_container["mime"] is not None
    assert input_container["mime"] == result["container"]["mime"]


def test_get_model(client):
    test_lift(client)
    result = client.execute(gql("{ info { model } }"))

    assert result["info"]["model"] is not None


def test_targets_from_step(client):
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
                    exact
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
    result = client.execute(q)

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
    assert binary_target["exact"]
    assert not lift_target["ready"]


def test_targets(client):
    q = gql(
        """
    {
        targets(pathspec: "") {
            name
        }
    }
    """
    )
    result = client.execute(q)

    names = [s["name"] for s in result["targets"]]
    assert "begin" in names


def test_produce(client):
    run_preliminary_analyses(client)

    q = gql(
        """
    {
        produce(step: "Lift", container: "module.ll", targetList: ":Root")
    }
    """
    )
    result = client.execute(q)

    assert "produce" in result, result["produce"] != ""


def test_produce_artifact(client):
    run_preliminary_analyses(client)

    q = gql(
        """
    {
        produceArtifacts(step: "Lift")
    }
    """
    )
    result = client.execute(q)

    assert "produceArtifacts" in result, result["produceArtifacts"] != ""


def test_function_endpoint(client):
    run_preliminary_analyses(client)

    client.execute(
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
    result = client.execute(q)

    first_function = next(t for t in result["container"]["targets"] if t["serialized"] != "")
    q = gql(
        """
    query function($param1: String!) {
        function(param1: $param1) {
            Isolate
        }
    }
    """
    )
    result = client.execute(q, {"param1": first_function["serialized"]})

    assert result["function"]["Isolate"] is not None


@mark.xfail(raises=Exception)
def test_analysis_kind_check(client):
    client.execute(
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

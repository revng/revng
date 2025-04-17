#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import atexit
import hashlib
import json
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from subprocess import DEVNULL, Popen
from threading import Event
from uuid import uuid4

import yaml

from revng.internal.api import initialize as capi_initialize
from revng.internal.api import shutdown as capi_shutdown
from revng.internal.api.syncing_manager import SyncingManager


def get_free_port():
    """Get an unbound port to spawn the S3 server on. Note that this is not
    failproof and might fail since there's a time gap between when the script
    is run and the server actually binds the socket"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def file_hash(path: str | Path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def resolve_path(root: Path, subpath: str):
    index_file = root / "index.yml._S3rver_object"
    assert index_file.is_file()
    with open(index_file) as f:
        index_data = yaml.safe_load(f)
    return root / f"{index_data[subpath]}._S3rver_object"


def main():
    input_binary = sys.argv[1]
    temporary_dir = sys.argv[2]
    temporary_dir_path = Path(temporary_dir)

    server_port = get_free_port()
    # Run the server with the specified port
    server_process = Popen(
        [
            "s3rver",
            "--directory",
            temporary_dir,
            "--address",
            "127.0.0.1",
            "--port",
            str(server_port),
            "--configure-bucket",
            "test",
        ],
        stdout=DEVNULL,
        stderr=DEVNULL,
    )
    atexit.register(server_process.kill)

    workdir = f"s3://S3RVER:S3RVER@region+127.0.0.1:{server_port}/test/project-test-dir"
    s3_root = temporary_dir_path / "test/project-test-dir"
    capi_initialize()
    atexit.register(capi_shutdown)

    save_event = Event()

    @contextmanager
    def wait_save():
        yield None
        save_event.wait()
        save_event.clear()

    with wait_save():
        manager = SyncingManager(workdir, save_hooks=(lambda x, y: save_event.set(),))

    with open(input_binary, "rb") as f:
        manager.set_input("input", f.read())

    with wait_save():
        manager.run_analyses_list("revng-initial-auto-analysis").unwrap()

    # Simple file check, this looks into the persistence directory of s3rver and
    # checks that the input file and the file in 'begin/input' have the same
    # contents. This is to guarantee that the S3 backend does not change the
    # contents of the file.
    s3_input_file_path = resolve_path(s3_root, "begin/input")
    assert file_hash(input_binary) == file_hash(s3_input_file_path)

    # Produce some stuff and then check that the file is actually saved
    step_name = "disassemble"
    container_name = "assembly.ptml.tar.gz"
    targets = manager.get_targets(step_name, container_name)
    manager.produce_target(step_name, [t.serialize() for t in targets], container_name)
    with wait_save():
        assert manager.save()

    s3_decompiled_path = resolve_path(s3_root, "disassemble/assembly.ptml.tar.gz")
    assert s3_decompiled_path.is_file() and s3_decompiled_path.stat().st_size > 0
    s3_decompiled_metadata = s3_decompiled_path.parent / (
        s3_decompiled_path.name.removesuffix("._S3rver_object") + "._S3rver_metadata.json"
    )
    with open(s3_decompiled_metadata) as f:
        metadata = json.load(f)
    assert metadata["content-encoding"] == "gzip"

    # Rename the first function in the model and check if it exists in the
    # saved model
    model = yaml.safe_load(manager.get_global("model.yml"))
    first_function = model["Functions"][0]
    unique_id = str(uuid4()).replace("-", "")
    diff = {
        "Path": f"/Functions/{first_function['Entry']}/Name",
        "Add": unique_id,
        "Remove": "",
    }
    if "Name" in first_function:
        diff["Remove"] = first_function["Name"]

    diff_content = yaml.safe_dump({"Changes": [diff]})
    with wait_save():
        manager.run_analysis(
            "initial",
            "apply-diff",
            {},
            {"apply-diff-global-name": "model.yml", "apply-diff-diff-content": diff_content},
        ).unwrap()

    model_path = resolve_path(s3_root, "context/model.yml")
    assert unique_id in model_path.read_text()
    # Also check that the partial save did not delete some file
    assert s3_decompiled_path.is_file() and s3_decompiled_path.stat().st_size > 0

    # Wait for the manager to stop its threads
    manager.stop()


if __name__ == "__main__":
    main()

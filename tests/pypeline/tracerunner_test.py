#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sqlite3
import sys
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

import yaml

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS objects(
    savepoint_id         INT NOT NULL,
    container_id         TEXT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    object_id            TEXT NOT NULL,
    content              BLOB NOT NULL,
    PRIMARY KEY (savepoint_id, container_id, configuration_hash, object_id)
) STRICT;
"""

PUT_QUERY = """
REPLACE INTO objects(savepoint_id, container_id, configuration_hash, object_id, content)
VALUES (?, ?, ?, ?, ?)
"""

GET_QUERY = """
SELECT object_id, content FROM objects
WHERE
  savepoint_id = ?
  AND container_id = ?
  AND configuration_hash = ?
  AND object_id IN ({id_list})"""


def run_test(tmp_path: Path):
    model_path = tmp_path / "model.yml"
    model_path.touch()

    db_path = tmp_path / "storage.db"
    object_1 = "/function/0x400000:Code_x86_64"
    object_2 = "/function/0x400020:Code_x86_64"
    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_TABLE)
        conn.execute(PUT_QUERY, (0, "main", "", object_1, b"foo"))
        conn.execute(PUT_QUERY, (0, "main", "", object_2, b"bar"))

    cmd = ["revng", "pypeline-trace-run"]
    cmd.extend([f"-load={e}" for e in sys.argv[2:]])
    cmd.extend([sys.argv[1], str(model_path.resolve()), str(db_path.resolve())])
    run(cmd, check=True)

    # Check the the pipe ran
    with sqlite3.connect(db_path) as conn:
        query = GET_QUERY.format(id_list=f"'{object_1}','{object_2}'")
        data = {r[0]: r[1] for r in conn.execute(query, [1, "main", ""])}
    assert data == {object_1: b"foofoo", object_2: b"bar"}

    # Check that the analysis ran
    yaml_model = yaml.safe_load(model_path.read_text())
    assert "foo.so" in yaml_model["ImportedLibraries"]


def main():
    with TemporaryDirectory() as tmp:
        run_test(Path(tmp))


if __name__ == "__main__":
    main()

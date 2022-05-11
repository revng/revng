#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from base64 import b64encode
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, List

from xdg import xdg_data_home


def clean_double_dict(dictionary: Dict[str, Dict[str, List]]):
    keys_to_delete = []
    for key in dictionary.keys():
        clean_dict(dictionary[key])
        if not dictionary[key]:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        dictionary.pop(key)


def clean_dict(dictionary: Dict[str, List]):
    keys_to_delete = []
    for key in dictionary.keys():
        if not dictionary[key]:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        dictionary.pop(key)


def clean_step_list(step_list: List):
    for step in step_list:
        clean_container_list(step["containers"])

    for step in step_list[:]:
        if len(step["containers"]) == 0:
            step_list.remove(step)


def clean_container_list(container_list: List):
    for container in container_list[:]:
        if len(container["targets"]) == 0:
            container_list.remove(container)


def str_to_snake_case(string: str) -> str:
    ret = []
    for idx, char in enumerate(string):
        if char.isupper():
            if (idx > 0 and string[idx - 1].isupper()) or idx == 0:
                ret.append(char.lower())
            else:
                ret += ["_", char.lower()]
        elif char == ".":
            ret.append("_")
        else:
            ret.append(char)
    return "".join(ret)


def b64e(string: str) -> str:
    ret = b64encode(string.encode("utf-8"))
    return ret.decode("utf-8")


def project_workdir() -> Path:
    data_dir = os.getenv("REVNG_DATA_DIR")
    project_id = os.getenv("REVNG_PROJECT_ID")

    if data_dir is None and project_id is None:
        workdir = Path(mkdtemp())
    elif data_dir is not None and project_id is None:
        workdir = Path(data_dir)
    elif project_id is not None:
        if data_dir is not None:
            workdir = Path(data_dir) / b64e(project_id)
        else:
            workdir = xdg_data_home() / "revng" / b64e(project_id)

    workdir.mkdir(parents=True, exist_ok=True)
    return workdir

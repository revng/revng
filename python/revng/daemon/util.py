#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
import os
from base64 import b64encode
from typing import Dict, List, Optional


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


def project_workdir() -> Optional[str]:
    return os.environ.get("REVNG_DATA_DIR")


def produce_serializer(input_: Dict[str, str | bytes]) -> str:
    return json.dumps(
        {
            key: (value if isinstance(value, str) else b64encode(value).decode("utf-8"))
            for key, value in input_.items()
        }
    )

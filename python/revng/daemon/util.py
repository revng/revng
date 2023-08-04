#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
import os
from base64 import b64encode
from typing import Dict, Optional


def project_workdir() -> Optional[str]:
    return os.environ.get("REVNG_DATA_DIR")


def produce_serializer(input_: Dict[str, str | bytes]) -> str:
    return json.dumps(
        {
            key: (value if isinstance(value, str) else b64encode(value).decode("utf-8"))
            for key, value in input_.items()
        }
    )

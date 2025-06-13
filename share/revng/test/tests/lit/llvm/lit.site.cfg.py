#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from pathlib import Path

current_directory = Path(__file__).parent.resolve()

# flake8: noqa: F821
config.my_src_root = current_directory  # type: ignore
config.my_obj_root = os.path.join(current_directory, "output")  # type: ignore
lit_config.load_config(config, os.path.join(current_directory, "lit.cfg.py"))  # type: ignore

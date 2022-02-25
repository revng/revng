#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Tests that revng.model is importable
import os
import sys
from pathlib import Path

build_dir = Path(os.environ["BUILD_DIR"])
sys.path.insert(0, f"{build_dir / 'lib' / 'python'}")

from revng import model

assert "Binary" in dir(model)

print("Test passed")

#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
set -euo pipefail

python -m pip install \
       --quiet \
       --compile \
       --no-index \
       --no-build-isolation \
       --ignore-installed \
       --no-deps \
       --root "$DESTDIR" \
       "$1"

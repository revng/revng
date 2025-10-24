#!/usr/bin/env bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

printf "Name\\n"

(
  for FILE in "$@"; do
    (
      if [[ "${FILE}" =~ ^.+\.bc$ ]]; then
        cat "${FILE}"

      elif [[ "${FILE}" =~ ^.+\.ll$ ]]; then
        llvm-as "${FILE}" -o -

      else
        echo "\"${FILE}\" does not have a supported llvm-module extension. Is it even a module?" 1>&2
        exit 1
      fi
    ) |

    llvm-nm - --format=just-symbols
  done
) | sort -u

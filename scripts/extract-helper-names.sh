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

    # TODO: remove `--defined-only` to pick up all the symbols instead of just
    # the symbol the helper modules explicitly contain.
    #
    # As of now, this is necessary because otherwise all the standard C library
    # functions helper binaries (both our and external) use populate the helper
    # namespace, leading to a heavy decrease in quality of decompiled binaries
    # that use c standard library (names like `malloc` and `free` (and so on)
    # are no longer allowed, no matter whether they are linked statically or
    # dynamically.
    #
    # Once everything within helper binaries is prefixed, this limitation is no
    # more and all the helper symbol names should be banned.
    #
    # The only downside of not doing so now is that *if* a helper using
    # a standard function call gets inlined, it is now calling a non-reserved
    # name, which might have a different meaning assigned to it by the model.
    llvm-nm - --defined-only --format=just-symbols
  done
) | sort -u

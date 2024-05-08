#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

set -euo pipefail

TEMP_FILE=$(mktemp --tmpdir --suffix=.tar.gz tmp.revng.GzipTarFileGeneratorTest.XXXXXXXXXX)
trap 'rm -f -- "$TEMP_FILE"' EXIT

"$1/test_gzip_tar_fileGenerator" foo foo2 bar bar2 > "$TEMP_FILE"
CONTENTS=$(tar -tf "$TEMP_FILE")

[[ $(wc -l <<< "$CONTENTS") -eq 2 ]]

grep -qF 'foo' <<< "$CONTENTS"
grep -qF 'bar' <<< "$CONTENTS"

[[ $(tar -xf "$TEMP_FILE" foo --to-stdout) = "foo2" ]]
[[ $(tar -xf "$TEMP_FILE" bar --to-stdout) = "bar2" ]]

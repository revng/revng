#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

set -euo pipefail

PACKAGE_DIR="$3.ts-package"
rm -rf "$PACKAGE_DIR"

mkdir "$PACKAGE_DIR"
BUILD_DIR="$(realpath "$PACKAGE_DIR")"
function cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  rm -rf "$BUILD_DIR"
}
trap cleanup SIGINT SIGTERM ERR EXIT

cd "$PACKAGE_DIR"

cp "$1/package-$3.json" package.json
cp "$1/tsconfig.json" tsconfig.json
cp "$1/tuple_tree.ts" tuple_tree.ts
cp ../lib/typescript/"$3".ts .
CHECKSUM=$(cat "$3.ts" "tuple_tree.ts" | sha1sum - | cut -d' ' -f1)
sed -i "s;##CHECKSUM##;$CHECKSUM;g" package.json
cp -rT "$2" node_modules
./node_modules/.bin/tsc -p .
npm pack --silent > /dev/null
cp "revng-$3-1."*.tgz ../"$3".ts.tgz

cd ..

if test -e lib64/node_modules; then
    echo "lib64/node_modules should not exist" > /dev/stderr
    exit 1
fi

npm --silent install --global --prefix=. "./$3.ts.tgz"

# Handle npm implementations using lib64
if test -e lib64/node_modules; then
    cp -ar lib64/node_modules lib/
    rm -rf lib64/node_modules
fi

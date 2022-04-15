#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

rm -rf model.ts-package

mkdir model.ts-package
BUILD_DIR="$(realpath model.ts-package)"
YARN_CACHE="$(mktemp -d)"
function cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  rm -rf "$YARN_CACHE"
  rm -rf "$BUILD_DIR"
}
trap cleanup SIGINT SIGTERM ERR EXIT

cd model.ts-package

cp "$1/package-model.json" package.json
cp "$1/tsconfig-model.json" tsconfig.json
cp "$1/tuple_tree.ts" tuple_tree.ts
cp ../lib/typescript/model.ts .
CHECKSUM=$(cat model.ts tuple_tree.ts | sha1sum - | cut -d' ' -f1)
sed -i "s;##CHECKSUM##;$CHECKSUM;g" package.json
cp -rT "$2" node_modules
./node_modules/.bin/tsc -p .
yarn -s pack
cp revng-model-v1.*.tgz ../model.ts.tgz

cd ..

# Use temporary cache folder to be extra sure that no caching is happening
yarn -s add --cache-folder "$YARN_CACHE" ./model.ts.tgz

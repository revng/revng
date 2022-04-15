#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

rm -rf model.ts-package

mkdir model.ts-package
BUILD_DIR="$(realpath model.ts-package)"
function cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  rm -rf "$BUILD_DIR"
}
trap cleanup SIGINT SIGTERM ERR EXIT

cd model.ts-package

cp "$1/package-model.json" package.json
cp "$1/tsconfig-model.json" tsconfig.json
cp "$1/tuple_tree.ts" tuple_tree.ts
cp ../lib/typescript/model.ts .
cp -rT "$2" node_modules
./node_modules/.bin/tsc -p .
yarn -s pack
cp revng-model-v1.0.0.tgz ../model.ts.tgz

cd ..

rm -rf model.ts-package

yarn -s add ./model.ts.tgz

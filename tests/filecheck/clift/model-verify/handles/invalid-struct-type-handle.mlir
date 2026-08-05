//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!t = !clift.struct<"/type-definition/1002-UnionDefinition" : size(1) {}>

// CHECK: a StructType with an invalid handle: '/type-definition/1002-UnionDefinition'
module attributes {clift.module, clift.test = !t} {}

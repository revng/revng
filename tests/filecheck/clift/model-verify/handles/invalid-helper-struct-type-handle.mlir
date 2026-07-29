//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng2 pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!t = !clift.union<"/helper-struct-type/foo" : {
  "/helper-struct-field/foo/field_0" : !clift.int<signed 4>
}>

// CHECK: a non-struct type with a HelperStructType handle: '/helper-struct-type/foo'
module attributes {clift.module, clift.test = !t} {}

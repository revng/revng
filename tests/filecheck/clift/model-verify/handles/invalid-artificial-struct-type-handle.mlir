//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!t = !clift.union<"/artificial-struct/foo" : {
  "/return-register/foo/rax_x86_64" : !clift.int<signed 4>
}>

// CHECK: a non-struct type with an ArtificialStruct handle: '/artificial-struct/foo'
module attributes {clift.module, clift.test = !t} {}

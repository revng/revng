//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe model-verify-clift %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!t = !clift.union<"/artificial-struct/foo" : {
  "/return-register/foo/rax_x86_64" : !clift.primitive<signed 4>
}>

// CHECK: non-struct type with ArtificialStruct handle: '/artificial-struct/foo'
module attributes {clift.module, clift.test = !t} {}

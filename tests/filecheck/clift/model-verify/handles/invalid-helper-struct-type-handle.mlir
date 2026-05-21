//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!t = !clift.union<"/helper-struct-type/foo" : {
  "/helper-struct-field/foo/field_0" : !clift.int<signed 4>
}>

// CHECK: a non-struct type with a HelperStructType handle: '/helper-struct-type/foo'
module attributes {clift.module, clift.test = !t} {}

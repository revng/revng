//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe model-verify-clift %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!t = !clift.union<"/helper-struct-type/foo" : {
  "/helper-struct-field/foo/field_0" : !clift.primitive<signed 4>
}>

// CHECK: non-struct type with HelperStructType handle: '/helper-struct-type/foo'
module attributes {clift.module, clift.test = !t} {}

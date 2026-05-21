//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!t = !clift.union<"/type-definition/1001-StructDefinition" : {
  "/struct-field/1001-StructDefinition/0" : !clift.int<signed 4>
}>

// CHECK: a UnionType with an invalid handle: '/type-definition/1001-StructDefinition'
module attributes {clift.module, clift.test = !t} {}

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!t = !clift.enum<"/type-definition/1001-StructDefinition" : !clift.int<signed 4> {
  "/enum-entry/1001-EnumDefinition/0" : 0
}>

// CHECK: an EnumType with an invalid handle: '/type-definition/1001-StructDefinition'
module attributes {clift.module, clift.test = !t} {}

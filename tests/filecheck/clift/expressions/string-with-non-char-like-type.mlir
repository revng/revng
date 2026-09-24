//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!char = !clift.enum<"/type-definition/1001-EnumDefinition" : !clift.int<signed 1> {
  "/enum-entry/1001-EnumDefinition/0" : 0
}>

// CHECK: result element type must be a primitive integer type or a C character type
clift.str "hello" : !clift.const<!clift.array<6 x !char>>

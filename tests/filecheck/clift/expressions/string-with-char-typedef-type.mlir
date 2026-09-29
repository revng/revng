//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!char = !clift.typedef<"/type-definition/1001-TypedefDefinition" : !clift.c_char<1>>

// CHECK: result element may not have typedef type with an underlying C character type
clift.str "hello" : !clift.const<!clift.array<6 x !char>>

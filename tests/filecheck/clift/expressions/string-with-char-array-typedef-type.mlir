//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!char_array = !clift.typedef<"/type-definition/1001-TypedefDefinition" : !clift.array<6 x !clift.c_char<1>>>

// CHECK: result may not have typedef type when the element type is a C character type
clift.str "hello" : !clift.const<!char_array>

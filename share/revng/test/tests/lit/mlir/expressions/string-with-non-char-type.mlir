//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!char$const = !clift.const<!clift.primitive<number 1>>
!char$const$ptr$const = !clift.const<!clift.ptr<8 to !char$const>>

// CHECK: result must have number8_t element type
clift.str "hello" : !clift.array<6 x !char$const$ptr$const>

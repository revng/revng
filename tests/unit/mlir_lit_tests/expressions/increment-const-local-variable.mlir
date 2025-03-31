//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$const = !clift.const<!clift.primitive<signed 4>>

%lvalue = clift.local !int32_t$const "x"

// CHECK: operand #0 must be modifiable
clift.inc %lvalue : !int32_t$const

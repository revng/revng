//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>
!int32_t$const = !clift.primitive<is_const = true, SignedKind 4>

%lvalue = clift.local !int32_t$const "x"

// CHECK: operand #0 must be modifiable
clift.assign %lvalue, %lvalue : !int32_t$const

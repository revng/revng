//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>

%i = clift.undef : !int32_t

// CHECK: requires either the operand or result to have floating point type
clift.cast<convert> %i : !int32_t -> !int32_t

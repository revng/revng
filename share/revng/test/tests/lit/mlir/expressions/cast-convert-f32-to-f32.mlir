//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!float32_t = !clift.primitive<float 4>

%f = clift.undef : !float32_t

// CHECK: result type cannot match the operand type
clift.cast<convert> %f : !float32_t -> !float32_t

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>

%value = clift.undef : !int32_t

// CHECK: result type must be wider than the argument type
clift.cast<extend> %value : !int32_t -> !int32_t

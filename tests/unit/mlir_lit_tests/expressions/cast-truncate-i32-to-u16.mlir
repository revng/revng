//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!uint16_t = !clift.primitive<unsigned 2>

%value = clift.undef : !int32_t

// CHECK: result and argument types must be equal in kind
clift.cast<truncate> %value : !int32_t -> !uint16_t

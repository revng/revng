//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int16_t = !clift.primitive<SignedKind 2>
!uint32_t = !clift.primitive<UnsignedKind 4>

%value = clift.undef : !int16_t

// CHECK: result and argument types must be equal in kind
clift.cast<extend> %value : !int16_t -> !uint32_t

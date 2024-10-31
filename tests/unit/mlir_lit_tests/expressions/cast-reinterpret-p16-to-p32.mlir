//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>
!int32_t$ptr32 = !clift.pointer<pointee_type = !int32_t, pointer_size = 4>
!int32_t$ptr64 = !clift.pointer<pointee_type = !int32_t, pointer_size = 8>

%p = clift.undef : !int32_t$ptr32

// CHECK: result and argument types must be equal in size
clift.cast<reinterpret> %p : !int32_t$ptr32 -> !int32_t$ptr64

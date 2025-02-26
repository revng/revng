//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!char$const = !clift.primitive<is_const = true, NumberKind 1>
!char$const$ptr$const = !clift.pointer<is_const = true, pointer_size = 8, pointee_type = !char$const>

// CHECK: result must have number8_t element type
clift.str "hello" : !clift.array<element_type = !char$const$ptr$const, elements_count = 6>

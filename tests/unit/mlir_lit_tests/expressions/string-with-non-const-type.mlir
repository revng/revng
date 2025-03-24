//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!char = !clift.primitive<number 1>

// CHECK: result must have const array type
clift.str "hello" : !clift.array<element_type = !char, elements_count = 6>

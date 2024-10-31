//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>
!uint32_t = !clift.primitive<UnsignedKind 4>

!f = !clift.defined<#clift.function<id = 1,
                                    name = "",
                                    return_type = !int32_t,
                                    argument_types = []>>

%f = clift.undef : !f

// CHECK: result type must match the return type of the function
"clift.call"(%f) : (!f) -> (!uint32_t)

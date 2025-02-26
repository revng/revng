//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>
!uint32_t = !clift.primitive<UnsignedKind 4>

!f = !clift.defined<#clift.function<unique_handle = "/model-type/1",
                                    name = "",
                                    return_type = !void,
                                    argument_types = [!int32_t]>>

%f = clift.undef : !f
%u = clift.undef : !uint32_t

// CHECK: argument types must match the parameter types
clift.call %f(%u : !uint32_t) : !f

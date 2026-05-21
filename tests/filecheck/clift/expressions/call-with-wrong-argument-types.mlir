//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !void(!int32_t)
>

%f = clift.undef : !f
%u = clift.undef : !uint32_t

// CHECK: argument types must match the parameter types
clift.call %f(%u : !uint32_t) : !f

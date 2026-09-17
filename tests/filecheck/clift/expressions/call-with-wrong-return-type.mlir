//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !int32_t()
>

%f = clift.undef : !clift.ptr<8 to !f>

// CHECK: result type must match the return type of the function
"clift.call"(%f) : (!clift.ptr<8 to !f>) -> (!uint32_t)

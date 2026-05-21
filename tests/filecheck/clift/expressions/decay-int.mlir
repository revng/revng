//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

%int = clift.undef : !int32_t

// CHECK: argument must have array or function type
clift.decay %int : !int32_t -> !int32_t$ptr

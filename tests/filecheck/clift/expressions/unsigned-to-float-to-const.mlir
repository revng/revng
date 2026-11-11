//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!uint32_t = !clift.int<unsigned 4>
!float32_t = !clift.float<4>

%value = clift.undef : !uint32_t

// CHECK: result #0 must be non-const floating point type
clift.utof %value : !uint32_t -> !clift.const<!float32_t>

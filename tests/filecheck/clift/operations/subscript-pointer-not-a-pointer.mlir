//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// The subscripted operand has to be a pointer. The custom form resolves it
// against that constraint while parsing, so the complaint comes from the
// parser rather than from the verifier.

!int32_t = !clift.int<signed 4>

%0 = clift.undef : !int32_t
%1 = clift.undef : !int32_t
// CHECK: custom op 'clift.subscript' 'pointer' must be pointer type
%2 = clift.subscript %0, %1 : (!int32_t, !int32_t)

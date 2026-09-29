//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s

!int8_t = !clift.int<signed 1>

// CHECK: clift.imm -128 : !int8_t
clift.imm -128 : !int8_t

// CHECK: clift.imm -1 : !int8_t
clift.imm -1 : !int8_t

// CHECK: clift.imm 0 : !int8_t
clift.imm 0 : !int8_t

// CHECK: clift.imm 127 : !int8_t
clift.imm 127 : !int8_t

// CHECK: clift.imm -1 : !int8_t
clift.imm 255 : !int8_t

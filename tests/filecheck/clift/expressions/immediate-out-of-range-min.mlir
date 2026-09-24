//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// CHECK: value out of range
clift.imm -129 : !clift.int<signed 1>

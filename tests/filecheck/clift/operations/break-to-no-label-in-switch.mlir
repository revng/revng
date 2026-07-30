//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>

// A plain break nested within a switch would break the switch, not the loop,
// so an operand-less break_to is invalid here.

clift.for body {
  clift.switch {
    %0 = clift.imm 0 : !int32_t
    clift.yield %0 : !int32_t
  } case 0 {
    // CHECK: clift.break_to with no target label may not be separated from its target loop by a switch
    clift.break_to
  }
}

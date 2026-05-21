//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt --verify-c %s 2>&1 | FileCheck %s

!void = !clift.void
!int16_t = !clift.int<signed 2>
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    // CHECK: not yielding the canonical boolean type
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 0 : !int32_t
      %2 = clift.eq %0, %1 : !int32_t -> !int16_t
      clift.yield %2 : !int16_t
    }
  }
}

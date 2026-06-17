//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt --verify-c %s 2>&1 | FileCheck %s

!void = !clift.void
!uint32_t = !clift.int<unsigned 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void(!uint32_t)
>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !uint32_t) {
    clift.expr {
      // CHECK: operand signedness does not match operation semantics
      %0 = clift.sar %arg0, %arg0 : (!uint32_t, !uint32_t)
      clift.yield %0 : !uint32_t
    }
  }
}

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --hoist-variable-initializers | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "f" : !void()
>

// Local is moved to just before its only user.

module attributes {clift.module} {
  clift.func @f<!f>() {
    // CHECK: %0 = clift.local : !int32_t = (%arg0) {
    %0 = clift.local : !int32_t
    // CHECK-NOT: clift.expr
    clift.expr {
      %1 = clift.assign %0, %0 : !int32_t
      // CHECK: clift.yield %arg0 : !int32_t
      clift.yield %1 : !int32_t
    }
    // CHECK: }
  }
}

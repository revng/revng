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
    // CHECK: %0 = clift.local : !int32_t attributes {
    // CHECK:   name = "var_1"
    // CHECK: }
    %0 = clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      name = "var_0"
    }
    // CHECK: %1 = clift.local : !int32_t = {
    %1 = clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/1",
      name = "var_1"
    }
    clift.expr {
      %2 = clift.assign %0, %1 : !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %2 : !int32_t
    }
    // CHECK: } attributes {
    // CHECK:   name = "var_0"
    // CHECK: }
  }
}

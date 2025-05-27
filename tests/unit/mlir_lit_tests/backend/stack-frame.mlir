//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>

!f = !clift.func<
  "/type-definition/1005-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001005(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001005:Code_x86_64"
  } {
    // CHECK: struct {{.*}}frame_1005 {
    // CHECK: int32_t a;
    // CHECK: };

    ^0: // Empty block to force function definition.
  }
  // CHECK: }
}

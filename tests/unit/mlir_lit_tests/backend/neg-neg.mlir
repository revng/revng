//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %x = clift.local !int32_t

    // CHECK: - -0;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.neg %0 : !int32_t
      %2 = clift.neg %1 : !int32_t
      clift.yield %2 : !int32_t
    }

    // CHECK: - -1;
    clift.expr {
      %0 = clift.imm -1 : !int32_t
      %1 = clift.neg %0 : !int32_t
      clift.yield %1 : !int32_t
    }

    // CHECK: - --x;
    clift.expr {
      %0 = clift.dec %x : !int32_t
      %1 = clift.neg %0 : !int32_t
      clift.yield %1 : !int32_t
    }

    // CHECK: ----x;
    clift.expr {
      %0 = clift.dec %x : !int32_t
      %1 = clift.dec %0 : !int32_t
      clift.yield %1 : !int32_t
    }
  }
  // CHECK: }
}

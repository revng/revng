//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

!int32_t = !clift.primitive<signed 4>
!int32_t$const = !clift.const<!clift.primitive<signed 4>>
!int32_t$const$ptr = !clift.ptr<8 to !int32_t$const>
!int32_t$const$ptr$const = !clift.const<!clift.ptr<8 to !int32_t$const>>
!int32_t$const$ptr$const$ptr = !clift.ptr<8 to !int32_t$const$ptr$const>
!int32_t$const$ptr$const$ptr$const = !clift.const<!clift.ptr<8 to !int32_t$const$ptr$const>>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: const int32_t var_0 = 0;
    %x = clift.local !int32_t$const = {
      %0 = clift.imm 0 : !int32_t
      clift.yield %0 : !int32_t
    }
    // CHECK: const int32_t *const var_1 = &var_0;
    %p = clift.local !int32_t$const$ptr$const = {
      %0 = clift.addressof %x : !int32_t$const$ptr
      clift.yield %0 : !int32_t$const$ptr
    }
    // CHECK: const int32_t *const *const var_2 = &var_1;
    %q = clift.local !int32_t$const$ptr$const$ptr$const = {
      %0 = clift.addressof %p : !int32_t$const$ptr$const$ptr
      clift.yield %0 : !int32_t$const$ptr$const$ptr
    }
  }
  // CHECK: }
}

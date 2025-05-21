//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!ptrdiff_t = !clift.primitive<signed 8>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %v0 = clift.local !int32_t
    %v1 = clift.local !int32_t

    // (0 + 1 - 2) * 3 / 4 % 5;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 2 : !int32_t
      %3 = clift.imm 3 : !int32_t
      %4 = clift.imm 4 : !int32_t
      %5 = clift.imm 5 : !int32_t

      %a = clift.add %0, %1 : !int32_t
      %b = clift.sub %a, %2 : !int32_t
      %c = clift.mul %b, %3 : !int32_t
      %d = clift.div %c, %4 : !int32_t
      %e = clift.rem %d, %5 : !int32_t

      clift.yield %e : !int32_t
    }

    // 0 + (1 - 2) * (3 / (4 % 5)
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 2 : !int32_t
      %3 = clift.imm 3 : !int32_t
      %4 = clift.imm 4 : !int32_t
      %5 = clift.imm 5 : !int32_t

      %b = clift.sub %1, %2 : !int32_t
      %e = clift.rem %4, %5 : !int32_t
      %d = clift.div %3, %e : !int32_t
      %c = clift.mul %b, %d : !int32_t
      %a = clift.add %0, %c : !int32_t

      clift.yield %a : !int32_t
    }

    // 0 % 1 / 2 * 3 + 4 - 5;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.imm 1 : !int32_t
      %2 = clift.imm 2 : !int32_t
      %3 = clift.imm 3 : !int32_t
      %4 = clift.imm 4 : !int32_t
      %5 = clift.imm 5 : !int32_t

      %a = clift.rem %0, %1 : !int32_t
      %b = clift.div %a, %2 : !int32_t
      %c = clift.mul %b, %3 : !int32_t
      %d = clift.sub %c, %4 : !int32_t
      %e = clift.add %d, %5 : !int32_t

      clift.yield %e : !int32_t
    }

    // CHECK: &var_0 + (&var_1 - &var_0);
    clift.expr {
      %0 = clift.addressof %v0 : !int32_t$ptr
      %1 = clift.addressof %v1 : !int32_t$ptr
      %2 = clift.ptr_diff %1, %0 : !int32_t$ptr -> !ptrdiff_t
      %4 = clift.addressof %v0 : !int32_t$ptr
      %3 = clift.ptr_add %4, %2 : (!int32_t$ptr, !ptrdiff_t)
      clift.yield %3 : !int32_t$ptr
    }

    // CHECK: &var_1 - (&var_1 - &var_0);
    clift.expr {
      %0 = clift.addressof %v0 : !int32_t$ptr
      %1 = clift.addressof %v1 : !int32_t$ptr
      %2 = clift.ptr_diff %1, %0 : !int32_t$ptr -> !ptrdiff_t
      %4 = clift.addressof %v1 : !int32_t$ptr
      %3 = clift.ptr_sub %4, %2 : (!int32_t$ptr, !ptrdiff_t)
      clift.yield %3 : !int32_t$ptr
    }
  }
  // CHECK: }
}

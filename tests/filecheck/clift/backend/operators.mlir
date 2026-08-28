//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-c=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>
!int32_t$p = !clift.ptr<8 to !int32_t>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: int32_t var_0;
    %x = clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      name = "var_0"
    }

    // CHECK: int32_t *var_1;
    %p = clift.local : !int32_t$p attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/1",
      name = "var_1"
    }

    // CHECK: -var_0;
    clift.expr {
      %r = clift.neg %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: ~var_0;
    clift.expr {
      %r = clift.bitnot %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: !(bool) var_0;
    clift.expr {
      %b = clift.test %x : !int32_t
      %r = clift.not %b
      clift.yield %r : !clift.bool
    }

    // CHECK: ++var_0;
    clift.expr {
      %r = clift.inc %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: --var_0;
    clift.expr {
      %r = clift.dec %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0++;
    clift.expr {
      %r = clift.post_inc %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0--;
    clift.expr {
      %r = clift.post_dec %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: &var_0;
    clift.expr {
      %r = clift.addressof %x : !int32_t$p
      clift.yield %r : !int32_t$p
    }

    // CHECK: *var_1;
    clift.expr {
      %r = clift.indirection %p : !int32_t$p
      clift.yield %r : !int32_t
    }

    // CHECK: var_0 + var_0;
    clift.expr {
      %r = clift.add %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0 - var_0;
    clift.expr {
      %r = clift.sub %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0 *var_0;
    clift.expr {
      %r = clift.mul %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0 / var_0;
    clift.expr {
      %r = clift.sdiv %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0 % var_0;
    clift.expr {
      %r = clift.srem %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: (bool) var_0 && (bool) var_0;
    clift.expr {
      %0 = clift.test %x : !int32_t
      %1 = clift.test %x : !int32_t
      %r = clift.and %0, %1
      clift.yield %r : !clift.bool
    }

    // CHECK: (bool) var_0 || (bool) var_0;
    clift.expr {
      %0 = clift.test %x : !int32_t
      %1 = clift.test %x : !int32_t
      %r = clift.or %0, %1
      clift.yield %r : !clift.bool
    }

    // CHECK: var_0 &var_0;
    clift.expr {
      %r = clift.bitand %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0 | var_0;
    clift.expr {
      %r = clift.bitor %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0 ^ var_0;
    clift.expr {
      %r = clift.bitxor %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0 << var_0;
    clift.expr {
      %r = clift.shl %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0 >> var_0;
    clift.expr {
      %r = clift.sar %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: var_0 == var_0;
    clift.expr {
      %r = clift.eq %x, %x : !int32_t
      clift.yield %r : !clift.bool
    }

    // CHECK: var_0 != var_0;
    clift.expr {
      %r = clift.ne %x, %x : !int32_t
      clift.yield %r : !clift.bool
    }

    // CHECK: var_0 < var_0;
    clift.expr {
      %r = clift.slt %x, %x : !int32_t
      clift.yield %r : !clift.bool
    }

    // CHECK: var_0 > var_0;
    clift.expr {
      %r = clift.sgt %x, %x : !int32_t
      clift.yield %r : !clift.bool
    }

    // CHECK: var_0 <= var_0;
    clift.expr {
      %r = clift.sle %x, %x : !int32_t
      clift.yield %r : !clift.bool
    }

    // CHECK: var_0 >= var_0;
    clift.expr {
      %r = clift.sge %x, %x : !int32_t
      clift.yield %r : !clift.bool
    }

    // CHECK: var_0, var_0;
    clift.expr {
      %r = clift.comma %x, %x : !int32_t, !int32_t
      clift.yield %r : !int32_t
    }
  }
  // CHECK: }
}

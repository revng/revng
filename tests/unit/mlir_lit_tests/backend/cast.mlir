//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>

!int64_t = !clift.primitive<signed 8>
!uint64_t = !clift.primitive<unsigned 8>

!uint64_t$ptr = !clift.ptr<8 to !uint64_t>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %x = clift.local !uint64_t

    // CHECK: (uint64_t)0;
    clift.expr {
      %0 = clift.imm 0 : !int64_t
      %1 = clift.cast<bitcast> %0 : !int64_t -> !uint64_t
      clift.yield %1 : !uint64_t
    }

    // CHECK: (uint64_t)&_var_0;
    clift.expr {
      %0 = clift.addressof %x : !uint64_t$ptr
      %1 = clift.cast<bitcast> %0 : !uint64_t$ptr -> !uint64_t
      clift.yield %1 : !uint64_t
    }
  }
  // CHECK: }
}

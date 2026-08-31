//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-c=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!void = !clift.void

!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>

!int64_t = !clift.int<signed 8>
!uint64_t = !clift.int<unsigned 8>

!int128_t = !clift.int<signed 16>
!uint128_t = !clift.int<unsigned 16>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

!my_enum = !clift.enum<
  "/type-definition/2001-EnumDefinition" as "my_enum" : !int64_t {
    "/enum-entry/2001-EnumDefinition/0" as "my_enum_0" : 0
  }
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: 0;
    clift.expr {
      %i = clift.imm 0 : !int32_t
      clift.yield %i : !int32_t
    }

    // CHECK: 0U;
    clift.expr {
      %u = clift.imm 0 : !uint32_t
      clift.yield %u : !uint32_t
    }

    // CHECK: 0L;
    clift.expr {
      %0 = clift.imm 0 : !int64_t
      clift.yield %0 : !int64_t
    }

    // CHECK: 0UL;
    clift.expr {
      %0 = clift.imm 0 : !uint64_t
      clift.yield %0 : !uint64_t
    }

    // CHECK: my_enum_0;
    clift.expr {
      %e = clift.imm 0 : !my_enum
      clift.yield %e : !my_enum
    }

    // CHECK: (my_enum) 1;
    clift.expr {
      %0 = clift.imm 1 : !int64_t
      %1 = clift.bitcast %0 : !int64_t -> !my_enum
      clift.yield %1 : !my_enum
    }
  }
  // CHECK: }
}

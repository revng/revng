//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1001",
  name = "",
  return_type = !void,
  argument_types = []>>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    unique_handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: while (0)
    clift.while {
      %0 = clift.imm 0 : !int32_t
      clift.yield %0 : !int32_t
    } {
      // CHECK: break;
      clift.loop_break
    }

    // CHECK: while (1)
    clift.while {
      %1 = clift.imm 1 : !int32_t
      clift.yield %1 : !int32_t
    } {
      // CHECK: continue;
      clift.loop_continue
    }

    // CHECK: while (2) {
    clift.while {
      %2 = clift.imm 2 : !int32_t
      clift.yield %2 : !int32_t
    } {
      // CHECK: 3;
      clift.expr {
        %3 = clift.imm 3 : !int32_t
        clift.yield %3 : !int32_t
      }
      // CHECK: 4;
      clift.expr {
        %4 = clift.imm 4 : !int32_t
        clift.yield %4 : !int32_t
      }
    }
    // CHECK: }
  }
  // CHECK: }
}

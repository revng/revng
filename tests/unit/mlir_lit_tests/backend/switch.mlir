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
    // CHECK: switch (0) {
    clift.switch {
      %0 = clift.imm 0 : !int32_t
      clift.yield %0 : !int32_t
    // CHECK: case 0: {
    } case 0 {
      // CHECK: 1;
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
      // CHECK: break;
      clift.switch_break
    // CHECK: }
    // CHECK: default:
    } default {
      // CHECK: 2;
      clift.expr {
        %2 = clift.imm 2 : !int32_t
        clift.yield %2 : !int32_t
      }
    }
    // CHECK: }
  }
  // CHECK: }
}

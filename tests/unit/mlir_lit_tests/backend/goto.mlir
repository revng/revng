//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !void,
  argument_types = []>>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    unique_handle = "/function/0x40001001:Code_x86_64"
  } {
    %exit = clift.make_label "exit"
    // CHECK: 0;
    clift.expr {
        %0 = clift.imm 0 : !int32_t
        clift.yield %0 : !int32_t
    }
    // CHECK: goto _label_0;
    clift.goto %exit
    // CHECK: _label_0: ;
    clift.assign_label %exit
  }
  // CHECK: }
}

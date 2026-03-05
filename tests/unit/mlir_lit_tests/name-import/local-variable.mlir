//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe import-descriptive-info %S/model.yml %s /dev/stdout | %revngcliftopt | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>

#loc1 = loc("/instruction/0x40001004:Code_x86_64/0x40001004:Code_x86_64/0x40001006:Code_x86_64")

module attributes { clift.module } {
  // CHECK: clift.func @fun_0x40001004<!fun_0x40001004_t>() -> !void
  // CHECK: attributes {
  // CHECK:   handle = "/function/0x40001004:Code_x86_64"
  // CHECK: }
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001004:Code_x86_64"
  } {
    // CHECK: %0 = clift.local : !int32_t attributes {
    // CHECK:   name = "var_0"
    // CHECK: }
    %0 = clift.local : !int32_t

    // CHECK: %1 = clift.local : !int32_t attributes {
    // CHECK: handle = "/local-variable/0x40001004:Code_x86_64/0"
    // CHECK: name = "my_local"
    // CHECK: }
    %1 = clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001004:Code_x86_64/0"
    }
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: clift.yield %1 : !int32_t
      clift.yield %1 : !int32_t loc(#loc1)
    // CHECK: }
    }
  }
}

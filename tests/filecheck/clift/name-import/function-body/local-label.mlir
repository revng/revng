//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng pipe import-descriptive-info %S/../model.yml %s /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!void = !clift.void

!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>

#loc1 = loc("/instruction/0x40001004:Code_x86_64/0x40001004:Code_x86_64/0x40001007:Code_x86_64")
#loc2 = loc("/instruction/0x40001004:Code_x86_64/0x40001004:Code_x86_64/0x40001008:Code_x86_64")

module attributes { clift.module } {
  // CHECK: clift.func @fun_0x40001004<!fun_0x40001004_t>() -> !void
  // CHECK: attributes {
  // CHECK:   handle = "/function/0x40001004:Code_x86_64"
  // CHECK: }
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001004:Code_x86_64"
  } {
    // CHECK: %0 = clift.make_label {
    // CHECK:   name = "label_0"
    // CHECK: }
    %0 = clift.make_label

    // CHECK: %1 = clift.make_label {
    // CHECK:   name = "my_label"
    // CHECK: }
    %1 = clift.make_label {
      handle = "/goto-label/0x40001004:Code_x86_64/0"
    }

    // CHECK: clift.assign_label %1
    clift.assign_label %1 loc(#loc1)

    // CHECK: clift.goto %1
    clift.goto %1 loc(#loc2)
  }
}

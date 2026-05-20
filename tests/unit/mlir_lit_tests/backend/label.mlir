//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s -o /dev/null | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @fun_0x40001001<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // Emits nothing.
    %label_0 = clift.make_label {
      handle = "/goto-label/0x40001001:Code_x86_64/0",
      name = "label_0"
    }

    // CHECK: int32_t var_0;
    %local_0 = clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      name = "var_0"
    }

    // CHECK: label_0:
    // CHECK-NOT: ;
    clift.assign_label %label_0

    // Emits nothing.
    clift.require %local_0 : !int32_t

    // Emits nothing.
    %label_1 = clift.make_label {
      handle = "/goto-label/0x40001001:Code_x86_64/1",
      name = "label_1"
    }

    // CHECK: label_1: ;
    clift.assign_label %label_1

    // Emits nothing.
    clift.require %local_0 : !int32_t

    // Emits nothing.
    %label_2 = clift.make_label {
      handle = "/goto-label/0x40001001:Code_x86_64/2",
      name = "label_2"
    }

    // CHECK: int32_t var_1;
    %local_1 = clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/1",
      name = "var_1"
    }

    // CHECK: label_2: ;
    clift.assign_label %label_2

    // Emits nothing.
    clift.require %local_0 : !int32_t
  }
  // CHECK: }
}

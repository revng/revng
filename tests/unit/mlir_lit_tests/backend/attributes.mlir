//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "fun_0x40001001_t" : !void(!int32_t)
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(int32_t x A1) A2 {
  clift.func @fun_0x40001001<!f>(%arg0 : !int32_t { clift.handle = "/cabi-argument/1001-CABIFunctionDefinition/0",
                                                    clift.name = "x",
                                                    clift.attributes = [#clift.attribute<"A1">] }) attributes {
    handle = "/function/0x40001001:Code_x86_64",
    clift.attributes = [
      #clift.attribute<"A2">
    ]
  } {
    // CHECK: int32_t y A3(argument);
    clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      clift.name = "y",
      clift.attributes = [
        #clift.attribute<"A3" : "/support-library/A3"("argument")>
      ]
    }
  }
  // CHECK: }
}

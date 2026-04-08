//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --emit-c %s | FileCheck %s
// RUN: %revngcliftopt --emit-c=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "fun_0x40001001_t" : !void(!int32_t)
>

module attributes {clift.module} {
  // CHECK: A2 void fun_0x40001001(int32_t x A1) {
  clift.func @fun_0x40001001<!f>(%arg0 : !int32_t { clift.handle = "/cabi-argument/1001-CABIFunctionDefinition/0",
                                                    clift.name = "x",
                                                    clift.c_attributes = [#clift.c_attribute<"A1">] }) attributes {
    handle = "/function/0x40001001:Code_x86_64",
    clift.c_attributes = [
      #clift.c_attribute<"A2">
    ]
  } {
    // CHECK: int32_t identifier_argument A3(argument);
    clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/0",
      name = "identifier_argument",
      clift.c_attributes = [
        #clift.c_attribute<"A3" : "/macro/A3" [#clift.identifier<"argument">]>
      ]
    }

    // CHECK: int32_t integer_argument A4(42);
    clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/1",
      name = "integer_argument",
      clift.c_attributes = [
        #clift.c_attribute<"A4" : "/macro/A4" [42]>
      ]
    }

    // CHECK: int32_t type_argument A5(int32_t);
    clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/2",
      name = "type_argument",
      clift.c_attributes = [
        #clift.c_attribute<"A5" : "/macro/A5" [!int32_t]>
      ]
    }

    // CHECK: int32_t multiple_arguments A6(argument, 42, int32_t);
    clift.local : !int32_t attributes {
      handle = "/local-variable/0x40001001:Code_x86_64/3",
      name = "multiple_arguments",
      clift.c_attributes = [
        #clift.c_attribute<"A6" : "/macro/A6" [#clift.identifier<"argument">, 42, !int32_t]>
      ]
    }
  }
  // CHECK: }
}

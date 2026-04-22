//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe import-descriptive-info %S/../0-import-types/CABIFunctionType.yml %s /dev/stdout | %revngcliftopt | FileCheck %s

!uint8_t = !clift.int<unsigned 1>
!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>

// CHECK: !cabifunction_0_ = !clift.func<
// CHECK:   "/type-definition/0-CABIFunctionDefinition" as "cabifunction_0" : !uint8_t(!uint16_t, !uint32_t)
// CHECK:   [
// CHECK:     #clift.c_attribute<"_ABI" : "/macro/_ABI"
// CHECK:       [
// CHECK:         #clift.identifier<"SystemV_x86_64">
// CHECK:       ]
// CHECK:     >
// CHECK:   ]
// CHECK:   comment "This comment is attached to the prototype (do not mistake it for\0Athe comment attached to the function itself!)"
// CHECK: >
// CHECK: module attributes {clift.module} {
// CHECK:   clift.func @my_commented_function<!cabifunction_0_>(
// CHECK:     !uint16_t {
// CHECK:       clift.comment = "This is what an argument comment looks like!",
// CHECK:       clift.handle = "/cabi-argument/0-CABIFunctionDefinition/0",
// CHECK:       clift.name = "argument_0"
// CHECK:     },
// CHECK:     !uint32_t {
// CHECK:       clift.comment = "And another one, for good measure!",
// CHECK:       clift.handle = "/cabi-argument/0-CABIFunctionDefinition/1",
// CHECK:       clift.name = "argument_1"
// CHECK:     }
// CHECK:   ) -> !uint8_t attributes {
// CHECK:     clift.c_attributes = [],
// CHECK:     clift.comment = "Unlike raw functions and all their complexities, CFTs are pretty\0Astraightforward as far as arguments are concerned. You just have the main\0Acomment, one for a return value, and one for each argument.",
// CHECK:     clift.return_value_comment = "And this is what return value one is like!",
// CHECK:     handle = "/function/0x4:Code_x86_64"
// CHECK:   }
// CHECK: }

!_type_definition_0_CABIFunctionDefinition = !clift.func<
  "/type-definition/0-CABIFunctionDefinition" : !uint8_t(!uint16_t, !uint32_t)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"SystemV_x86_64">]>
  ]
>
module attributes {clift.module} {
  clift.func @"0x4:Code_x86_64"<!_type_definition_0_CABIFunctionDefinition>(!uint16_t, !uint32_t) -> !uint8_t
  attributes {clift.c_attributes = [], handle = "/function/0x4:Code_x86_64"}
}

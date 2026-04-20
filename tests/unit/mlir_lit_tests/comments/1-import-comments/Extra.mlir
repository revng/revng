//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe import-descriptive-info %S/../0-import-types/Extra.yml %s /dev/stdout | %revngcliftopt | FileCheck %s

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
// CHECK: >

// CHECK: !cabifunction_1_ = !clift.func<
// CHECK:   "/type-definition/1-CABIFunctionDefinition" as "cabifunction_1" : !uint8_t(!uint16_t, !uint32_t)
// CHECK:   [
// CHECK:     #clift.c_attribute<"_ABI" : "/macro/_ABI"
// CHECK:       [
// CHECK:         #clift.identifier<"SystemV_x86_64">
// CHECK:       ]
// CHECK:     >
// CHECK:   ]
// CHECK: >

// CHECK: !cabifunction_2_ = !clift.func<
// CHECK:   "/type-definition/2-CABIFunctionDefinition" as "cabifunction_2" : !uint8_t(!uint16_t, !uint32_t)
// CHECK:   [
// CHECK:     #clift.c_attribute<"_ABI" : "/macro/_ABI"
// CHECK:       [
// CHECK:         #clift.identifier<"SystemV_x86_64">
// CHECK:       ]
// CHECK:     >
// CHECK:   ]
// CHECK: >

// CHECK: module attributes {clift.module} {

// CHECK:   clift.func @single_comment_function<!cabifunction_0_>(
// CHECK:     !uint16_t {
// CHECK:       clift.handle = "/cabi-argument/0-CABIFunctionDefinition/0",
// CHECK:       clift.name = "argument_0"
// CHECK:     },
// CHECK:     !uint32_t {
// CHECK:       clift.handle = "/cabi-argument/0-CABIFunctionDefinition/1",
// CHECK:       clift.name = "argument_1"
// CHECK:     }
// CHECK:   ) -> !uint8_t attributes {
// CHECK:     clift.c_attributes = [],
// CHECK:     clift.comment = "This function only has the function comment, with nothing attached\0Ato the prototype!"
// CHECK:     handle = "/function/0x4:Code_x86_64"
// CHECK:   }

// CHECK:   clift.func @argument_comment_function<!cabifunction_1_>(
// CHECK:     !uint16_t {
// CHECK:       clift.handle = "/cabi-argument/1-CABIFunctionDefinition/0",
// CHECK:       clift.name = "argument_0"
// CHECK:     },
// CHECK:     !uint32_t {
// CHECK:       clift.comment = "This function only has an argument comment!",
// CHECK:       clift.handle = "/cabi-argument/1-CABIFunctionDefinition/1",
// CHECK:       clift.name = "argument_1"
// CHECK:     }
// CHECK:   ) -> !uint8_t attributes {
// CHECK:     clift.c_attributes = [],
// CHECK:     clift.comment = "",
// CHECK:     handle = "/function/0x8:Code_x86_64"
// CHECK:   }

// CHECK:   clift.func @return_value_comment_function<!cabifunction_2_>(
// CHECK:     !uint16_t {
// CHECK:       clift.handle = "/cabi-argument/2-CABIFunctionDefinition/0",
// CHECK:       clift.name = "argument_0"
// CHECK:     },
// CHECK:     !uint32_t {
// CHECK:       clift.handle = "/cabi-argument/2-CABIFunctionDefinition/1",
// CHECK:       clift.name = "argument_1"
// CHECK:     }
// CHECK:   ) -> !uint8_t attributes {
// CHECK:     clift.c_attributes = [],
// CHECK:     clift.comment = "",
// CHECK:     clift.return_value_comment = "This function only has a return value comment!",
// CHECK:     handle = "/function/0xb:Code_x86_64"
// CHECK:   }
// CHECK: }

!_type_definition_0_CABIFunctionDefinition = !clift.func<
  "/type-definition/0-CABIFunctionDefinition" : !uint8_t(!uint16_t, !uint32_t)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"SystemV_x86_64">]>
  ]
>

!_type_definition_1_CABIFunctionDefinition = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" : !uint8_t(!uint16_t, !uint32_t)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"SystemV_x86_64">]>
  ]
>

!_type_definition_2_CABIFunctionDefinition = !clift.func<
  "/type-definition/2-CABIFunctionDefinition" : !uint8_t(!uint16_t, !uint32_t)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"SystemV_x86_64">]>
  ]
>

module attributes {clift.module} {
  clift.func @"0x4:Code_x86_64"<!_type_definition_0_CABIFunctionDefinition>(!uint16_t, !uint32_t) -> !uint8_t
  attributes {clift.c_attributes = [], handle = "/function/0x4:Code_x86_64"}

  clift.func @"0x8:Code_x86_64"<!_type_definition_1_CABIFunctionDefinition>(!uint16_t, !uint32_t) -> !uint8_t
  attributes {clift.c_attributes = [], handle = "/function/0x8:Code_x86_64"}

  clift.func @"0xb:Code_x86_64"<!_type_definition_2_CABIFunctionDefinition>(!uint16_t, !uint32_t) -> !uint8_t
  attributes {clift.c_attributes = [], handle = "/function/0xb:Code_x86_64"}
}

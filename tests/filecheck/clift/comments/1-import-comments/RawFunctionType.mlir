//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng2 pipeline run-pipe import-descriptive-info %S/../0-import-types/RawFunctionType.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!uint64_t = !clift.int<unsigned 8>

// CHECK: !my_struct = !clift.struct<
// CHECK:   "/type-definition/1-StructDefinition" as "my_struct" : size(8) {
// CHECK:     "/struct-field/1-StructDefinition/0" as "offset_0" : offset(0) !uint64_t
// CHECK:     comment "And since this is a struct argument, it, obviously, can have comments\0Aattached! And with how relevant this comment is for the prototype,\0Awe should probably find a way to display it *on* the said prototype!"
// CHECK:   }
// CHECK:   comment "Not all arguments are passed in registers! Sometimes there's also a struct!\0A\0AWhich is the revng-way of representing the stack offsets!"
// CHECK: >
// CHECK: !rawfunction_0_ = !clift.func<
// CHECK:   "/type-definition/0-RawFunctionDefinition" as "rawfunction_0" : !uint64_t(!uint64_t, !my_struct)
// CHECK:   [
// CHECK:     #clift.c_attribute<"_ABI" : "/macro/_ABI"
// CHECK:       [
// CHECK:         #clift.identifier<"raw_x86_64">
// CHECK:       ]
// CHECK:     >
// CHECK:   ]
// CHECK:   comment "This comment is attached to the prototype (do not mistake it for\0Athe comment attached to the function itself!)"
// CHECK: >
// CHECK: module attributes {clift.module} {
// CHECK:   clift.func @my_commented_function<!rawfunction_0_>(
// CHECK:     !uint64_t {
// CHECK:       clift.c_attributes =
// CHECK:         [
// CHECK:           #clift.c_attribute<
// CHECK:             "_REG" : "/macro/_REG"
// CHECK:             [
// CHECK:               #clift.identifier<"rax_x86_64">
// CHECK:             ]
// CHECK:           >
// CHECK:         ],
// CHECK:       clift.comment = "Let's be brief here,\0Ato make space for that ugly return value comment!",
// CHECK:       clift.handle = "/raw-argument/0-RawFunctionDefinition/rax_x86_64",
// CHECK:       clift.name = "register_rax"
// CHECK:     },
// CHECK:     !my_struct {
// CHECK:       clift.c_attributes =
// CHECK:         [
// CHECK:           #clift.c_attribute<"_STACK" : "/macro/_STACK">
// CHECK:         ],
// CHECK:       clift.handle = "/raw-stack-arguments/0-RawFunctionDefinition",
// CHECK:       clift.name = "stack_arguments"
// CHECK:     }
// CHECK:   ) -> !uint64_t attributes {
// CHECK:     clift.c_attributes = [],
// CHECK:     clift.comment = "Unlike many other structures with simpler comments, function types emit\0Atheirs as doxygen! They include specific sections for arguments! And\0Areturn values!",
// CHECK:     clift.return_value_comment = "Even though this function only returns one register, who's to say that\0Aregister is not interesting enough to write a really long essay about it?!\0A\0AWith multiple line breaks,\0A\0Aand non-trivial formatting too!\0A```cpp\0A // Meta comment: comment within a comment\0A // ```cpp\0A // // Could be within another comment too!!!\0A // ```\0A```\0A\0ALet's see how well this will be handled!!",
// CHECK:     handle = "/function/0x4:Code_x86_64"
// CHECK:   }
// CHECK: }

!_type_definition_1_StructDefinition = !clift.struct<
  "/type-definition/1-StructDefinition" : size(8) {
    "/struct-field/1-StructDefinition/0" : offset(0) !uint64_t
  }
>
!_type_definition_0_RawFunctionDefinition = !clift.func<
  "/type-definition/0-RawFunctionDefinition" : !uint64_t(!uint64_t, !_type_definition_1_StructDefinition)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_x86_64">]>
  ]
>

module attributes {clift.module} {
  clift.func @"0x4:Code_x86_64"<!_type_definition_0_RawFunctionDefinition>(
    !uint64_t,
    !_type_definition_1_StructDefinition
  ) -> !uint64_t
  attributes {clift.c_attributes = [], handle = "/function/0x4:Code_x86_64"}
}

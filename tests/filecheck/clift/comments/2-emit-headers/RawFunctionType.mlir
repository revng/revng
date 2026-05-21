//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-type-and-global-header %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-type-and-global-header=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!uint64_t = !clift.int<unsigned 8>

// CHECK: /// This comment is attached to the prototype (do not mistake it for
// CHECK: /// the comment attached to the function itself!)
// CHECK: typedef _ABI(raw_x86_64) uint64_t rawfunction_0(uint64_t, my_struct);
//
// CHECK: /// Not all arguments are passed in registers! Sometimes there's also a struct!
// CHECK: ///
// CHECK: /// Which is the revng-way of representing the stack offsets!
// CHECK: struct _PACKED _SIZE(8) my_struct {
//
// CHECK:   /// And since this is a struct argument, it, obviously, can have comments
// CHECK:   /// attached! And with how relevant this comment is for the prototype,
// CHECK:   /// we should probably find a way to display it *on* the said prototype!
// CHECK:   uint64_t offset_0;
// CHECK: };
//
// CHECK: /
// CHECK: / Functions
// CHECK: /
//
// CHECK: /// Unlike many other structures with simpler comments, function types emit
// CHECK: /// theirs as doxygen! They include specific sections for arguments! And
// CHECK: /// return values!
// CHECK: ///
// CHECK: /// \param register_rax Let's be brief here,
// CHECK: ///                     to make space for that ugly return value comment!
// CHECK: ///
// CHECK: /// \returns Even though this function only returns one register, who's to say that
// CHECK: ///          register is not interesting enough to write a really long essay about it?!
// CHECK: ///
// CHECK: ///          With multiple line breaks,
// CHECK: ///
// CHECK: ///          and non-trivial formatting too!
// CHECK: ///          ```cpp
// CHECK: ///            // Meta comment: comment within a comment
// CHECK: ///            // ```cpp
// CHECK: ///            //   // Could be within another comment too!!!
// CHECK: ///            // ```
// CHECK: ///          ```
// CHECK: ///
// CHECK: ///          Let's see how well this will be handled!!
// CHECK: _ABI(raw_x86_64) uint64_t my_commented_function(uint64_t register_rax _REG(rax_x86_64), my_struct stack_arguments _STACK);

!my_struct = !clift.struct<
  "/type-definition/1-StructDefinition" as "my_struct" : size(8) {
    "/struct-field/1-StructDefinition/0" as "offset_0" : offset(0) !uint64_t
    comment "And since this is a struct argument, it, obviously, can have comments\0Aattached! And with how relevant this comment is for the prototype,\0Awe should probably find a way to display it *on* the said prototype!"
  }
  comment "Not all arguments are passed in registers! Sometimes there's also a struct!\0A\0AWhich is the revng-way of representing the stack offsets!"
>
!rawfunction_0_ = !clift.func<
  "/type-definition/0-RawFunctionDefinition" as "rawfunction_0" : !uint64_t(!uint64_t, !my_struct)
  [ #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_x86_64">]> ]
  comment "This comment is attached to the prototype (do not mistake it for\0Athe comment attached to the function itself!)"
>
module attributes {clift.module} {
  clift.func @my_commented_function<!rawfunction_0_>(
    !uint64_t {
      clift.c_attributes = [#clift.c_attribute<"_REG" : "/macro/_REG" [#clift.identifier<"rax_x86_64">]>],
      clift.comment = "Let's be brief here,\0Ato make space for that ugly return value comment!",
      clift.handle = "/raw-argument/0-RawFunctionDefinition/rax_x86_64",
      clift.name = "register_rax"
    },
    !my_struct {
      clift.c_attributes = [#clift.c_attribute<"_STACK" : "/macro/_STACK">],
      clift.handle = "/raw-stack-arguments/0-RawFunctionDefinition",
      clift.name = "stack_arguments"
    }
  ) -> !uint64_t attributes {
    clift.c_attributes = [],
    clift.comment = "Unlike many other structures with simpler comments, function types emit\0Atheirs as doxygen! They include specific sections for arguments! And\0Areturn values!",
    clift.return_value_comment = "Even though this function only returns one register, who's to say that\0Aregister is not interesting enough to write a really long essay about it?!\0A\0AWith multiple line breaks,\0A\0Aand non-trivial formatting too!\0A```cpp\0A  // Meta comment: comment within a comment\0A  // ```cpp\0A  //   // Could be within another comment too!!!\0A  // ```\0A```\0A\0ALet's see how well this will be handled!!",
    handle = "/function/0x4:Code_x86_64"
  }
}

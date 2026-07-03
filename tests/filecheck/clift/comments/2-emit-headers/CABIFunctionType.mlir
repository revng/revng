//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt --emit-type-and-global-header %s -o /dev/null | FileCheck %s
// RUN: %root/bin/revng clift-opt --emit-type-and-global-header=ptml %s -o /dev/null | %root/bin/revng ptml | FileCheck %s

!uint8_t = !clift.int<unsigned 1>
!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>

// CHECK: //
// CHECK: // Types
// CHECK: //
//
// CHECK: typedef _ABI(SystemV_x86_64)
// CHECK: uint8_t cabifunction_0(uint16_t, uint32_t);
//
// CHECK: //
// CHECK: // Functions
// CHECK: //
//
// CHECK: /// Unlike raw functions and all their complexities, CFTs are pretty
// CHECK: /// straightforward as far as arguments are concerned. You just have the main
// CHECK: /// comment, one for a return value, and one for each argument.
// CHECK: ///
// CHECK: /// \param argument_0 This is what an argument comment looks like!
// CHECK: /// \param argument_1 And another one, for good measure!
// CHECK: ///
// CHECK: /// \returns This comment is attached to the prototype (do not mistake it for
// CHECK: /// the comment attached to the function itself!)
// CHECK: _ABI(SystemV_x86_64)
// CHECK: uint8_t my_commented_function(uint16_t argument_0, uint32_t argument_1);

!cabifunction_0_ = !clift.func<
  "/type-definition/0-CABIFunctionDefinition" as "cabifunction_0" : !uint8_t(!uint16_t, !uint32_t)
  [ #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"SystemV_x86_64">]> ]
  comment "This comment is attached to the prototype (do not mistake it for\0Athe comment attached to the function itself!)"
>
module attributes {clift.module} {
  clift.func @my_commented_function<!cabifunction_0_>(
    !uint16_t {
      clift.comment = "This is what an argument comment looks like!",
      clift.handle = "/cabi-argument/0-CABIFunctionDefinition/0",
      clift.name = "argument_0"
    },
    !uint32_t {
      clift.comment = "And another one, for good measure!",
      clift.handle = "/cabi-argument/0-CABIFunctionDefinition/1",
      clift.name = "argument_1"
    }
  ) -> !uint8_t attributes {
    clift.c_attributes = [],
    clift.comment = "Unlike raw functions and all their complexities, CFTs are pretty\0Astraightforward as far as arguments are concerned. You just have the main\0Acomment, one for a return value, and one for each argument.",
    clift.return_value_comment = "This comment is attached to the prototype (do not mistake it for\0Athe comment attached to the function itself!)",
    handle = "/function/0x4:Code_x86_64"
  }
}

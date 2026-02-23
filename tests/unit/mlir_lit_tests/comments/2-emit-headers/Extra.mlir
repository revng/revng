//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: ../%revngpipe emit-type-and-global-header /dev/null %s /dev/null /dev/stdout | ../%revngptml | FileCheck %s

!uint8_t = !clift.primitive<unsigned 1>
!uint16_t = !clift.primitive<unsigned 2>
!uint32_t = !clift.primitive<unsigned 4>

// CHECK: /// This function only has the function comment, with nothing attached
// CHECK: /// to the prototype!
// CHECK: _ABI(SystemV_x86_64) uint8_t single_comment_function(uint16_t argument_0, uint32_t argument_1);
//
// CHECK: /// \param argument_1 This function only has an argument comment!
// CHECK: _ABI(SystemV_x86_64) uint8_t argument_comment_function(uint16_t argument_0, uint32_t argument_1);
//
// CHECK: /// \returns This function only has a return value comment!
// CHECK: _ABI(SystemV_x86_64) uint8_t return_value_comment_function(uint16_t argument_0, uint32_t argument_1);

!cabifunction_0_ = !clift.func<
  "/type-definition/0-CABIFunctionDefinition" as "cabifunction_0" : !uint8_t(!uint16_t, !uint32_t)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI"(
      identifier "SystemV_x86_64" : "/macro-argument/_ABI/SystemV_x86_64"
    )>
  ]
>

!cabifunction_1_ = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" as "cabifunction_0" : !uint8_t(!uint16_t, !uint32_t)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI"(
      identifier "SystemV_x86_64" : "/macro-argument/_ABI/SystemV_x86_64"
    )>
  ]
>

!cabifunction_2_ = !clift.func<
  "/type-definition/2-CABIFunctionDefinition" as "cabifunction_0" : !uint8_t(!uint16_t, !uint32_t)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI"(
      identifier "SystemV_x86_64" : "/macro-argument/_ABI/SystemV_x86_64"
    )>
  ]
  return_value_comment "This function only has a return value comment!"
>

module attributes {clift.module} {

  clift.func @single_comment_function<!cabifunction_0_>(
    !uint16_t {
      clift.comment = "",
      clift.handle = "/cabi-argument/0-CABIFunctionDefinition/0",
      clift.name = "argument_0"
    },
    !uint32_t {
      clift.comment = "",
      clift.handle = "/cabi-argument/0-CABIFunctionDefinition/1",
      clift.name = "argument_1"
    }
  ) -> !uint8_t attributes {
    clift.c_attributes = [],
    clift.comment = "This function only has the function comment, with nothing attached\0Ato the prototype!",
    handle = "/function/0x4:Code_x86_64"
  }

  clift.func @argument_comment_function<!cabifunction_1_>(
    !uint16_t {
      clift.comment = "",
      clift.handle = "/cabi-argument/1-CABIFunctionDefinition/0",
      clift.name = "argument_0"
    },
    !uint32_t {
      clift.comment = "This function only has an argument comment!",
      clift.handle = "/cabi-argument/1-CABIFunctionDefinition/1",
      clift.name = "argument_1"
    }
  ) -> !uint8_t attributes {
    clift.c_attributes = [],
    clift.comment = "",
    handle = "/function/0x8:Code_x86_64"
  }

  clift.func @return_value_comment_function<!cabifunction_2_>(
    !uint16_t {
      clift.comment = "",
      clift.handle = "/cabi-argument/2-CABIFunctionDefinition/0",
      clift.name = "argument_0"
    },
    !uint32_t {
      clift.comment = "",
      clift.handle = "/cabi-argument/2-CABIFunctionDefinition/1",
      clift.name = "argument_1"
    }
  ) -> !uint8_t attributes {
    clift.c_attributes = [],
    clift.comment = "This function only has the function comment, with nothing attached\0Ato the prototype!",
    handle = "/function/0xb:Code_x86_64"
  }

}

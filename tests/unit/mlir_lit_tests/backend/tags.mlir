//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="model=%S/model.yml" | FileCheck %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>

!s = !clift.defined<#clift.struct<
  id = 2002,
  name = "",
  size = 4,
  fields = [
    <
      name = "",
      offset = 0,
      type = !int32_t
    >,
    <
      name = "",
      offset = 4,
      type = !int32_t
    >
  ]>>

!g = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !void,
  argument_types = []>>

!f = !clift.defined<#clift.function<
  id = 1004,
  name = "",
  return_type = !void,
  argument_types = [!int32_t]>>

clift.module {
  clift.global !int32_t @seg_external {
    unique_handle = "/segment/0x40002001:Generic64-4"
  }

  clift.func @fun_external<!g>() attributes {
    unique_handle = "/function/0x40001001:Code_x86_64"
  }

  clift.func @fun_imported<!g>() attributes {
    unique_handle = "/dynamic-function/imported"
  }

  // CHECK: fun_0x40001004
  // CHECK: {
  clift.func @f<!f>(%x : !int32_t) attributes {
    unique_handle = "/function/0x40001004:Code_x86_64"
  } {
    // CHECK: <span data-token="c.type" data-action-context-location="/primitive/int32_t" data-location-references="/primitive/int32_t">
    // CHECK: int32_t
    // CHECK: </span>
    // CHECK: <span data-location-definition="/local-variable/0x40001004:Code_x86_64/_var_0" data-token="c.variable">
    // CHECK: _var_0
    // CHECK: </span>
    // CHECK: <span data-token="c.constant">
    // CHECK: 0
    // CHECK: </span>
    // CHECK: ;
    %y = clift.local !int32_t "y" = {
      %0 = clift.imm 0 : !int32_t
      clift.yield %0 : !int32_t
    }

    // CHECK: <span data-token="c.function_parameter" data-location-references="/local-variable/0x40001004:Code_x86_64/x">
    // CHECK: x
    // CHECK: </span>
    // CHECK: ;
    clift.expr {
      clift.yield %x : !int32_t
    }

    // CHECK: <span data-token="c.variable" data-location-references="/local-variable/0x40001004:Code_x86_64/_var_0">
    // CHECK: _var_0
    // CHECK: </span>
    // CHECK: ;
    clift.expr {
      clift.yield %y : !int32_t
    }

    // CHECK: <span data-token="c.variable" data-action-context-location="/segment/0x40002001:Generic64-4" data-location-references="/segment/0x40002001:Generic64-4">
    // CHECK: seg_0x40002001
    // CHECK: </span>
    // CHECK: ;
    clift.expr {
      %r = clift.use @seg_external : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: <span data-token="c.function" data-action-context-location="/function/0x40001001:Code_x86_64" data-location-references="/function/0x40001001:Code_x86_64">
    // CHECK: fun_0x40001001
    // CHECK: </span>
    // CHECK: ;
    clift.expr {
      %r = clift.use @fun_external : !g
      clift.yield %r : !g
    }

    // CHECK: <span data-token="c.function" data-action-context-location="/dynamic-function/imported" data-location-references="/dynamic-function/imported">
    // CHECK: imported
    // CHECK: </span>
    // CHECK: ;
    clift.expr {
      %r = clift.use @fun_imported : !g
      clift.yield %r : !g
    }

    // CHECK: <span data-token="c.keyword">
    // CHECK: struct
    // CHECK: </span>
    // CHECK: <span data-token="c.type" data-action-context-location="/type-definition/2002-StructDefinition" data-location-references="/type-definition/2002-StructDefinition">
    // CHECK: my_struct
    // CHECK: </span>
    // CHECK: <span data-location-definition="/local-variable/0x40001004:Code_x86_64/_var_1" data-token="c.variable">
    // CHECK: _var_1
    // CHECK: </span>
    // CHECK: ;
    %s = clift.local !s "s"

    // CHECK: <span data-token="c.variable" data-location-references="/local-variable/0x40001004:Code_x86_64/_var_1">
    // CHECK: _var_1
    // CHECK: </span>
    // CHECK: <span data-token="c.operator">
    // CHECK: .
    // CHECK: </span>
    // CHECK: <span data-token="c.field" data-action-context-location="/struct-field/2002-StructDefinition/0" data-location-references="/struct-field/2002-StructDefinition/0">
    // CHECK: x
    // CHECK: </span>
    // CHECK: ;
    clift.expr {
      %r = clift.access<0> %s : !s -> !int32_t
      clift.yield %r : !int32_t
    }

    %label = clift.make_label "label"

    // CHECK: <span data-location-definition="/goto-label/0x40001004:Code_x86_64/_label_0" data-token="c.goto_label">
    // CHECK: _label_0
    // CHECK: </span>
    // CHECK: :
    clift.assign_label %label

    // CHECK: <span data-token="c.keyword">
    // CHECK: goto
    // CHECK: </span>
    // CHECK: <span data-token="c.goto_label" data-location-references="/goto-label/0x40001004:Code_x86_64/_label_0">
    // CHECK: _label_0
    // CHECK: </span>
    // CHECK: ;
    clift.goto %label
  }
  // CHECK: }
}

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.defined<#clift.function<
  unique_handle = "/model-type/1001",
  name = "",
  return_type = !void,
  argument_types = []>>

!int32_t = !clift.primitive<signed 4>
!int32_t$const = !clift.primitive<const signed 4>
!int32_t$const$ptr = !clift.pointer<pointer_size = 8, pointee_type = !int32_t$const>
!int32_t$const$ptr$const = !clift.pointer<is_const = true, pointer_size = 8, pointee_type = !int32_t$const>
!int32_t$const$ptr$const$ptr = !clift.pointer<pointer_size = 8, pointee_type = !int32_t$const$ptr$const>
!int32_t$const$ptr$const$ptr$const = !clift.pointer<is_const = true, pointer_size = 8, pointee_type = !int32_t$const$ptr$const>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    unique_handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: const int32_t _var_0 = 0;
    %x = clift.local !int32_t$const "x" = {
      %0 = clift.imm 0 : !int32_t
      clift.yield %0 : !int32_t
    }
    // CHECK: const int32_t *const _var_1 = &_var_0;
    %p = clift.local !int32_t$const$ptr$const "p" = {
      %0 = clift.addressof %x : !int32_t$const$ptr
      clift.yield %0 : !int32_t$const$ptr
    }
    // CHECK: const int32_t *const *const _var_2 = &_var_1;
    %q = clift.local !int32_t$const$ptr$const$ptr$const "q" = {
      %0 = clift.addressof %p : !int32_t$const$ptr$const$ptr
      clift.yield %0 : !int32_t$const$ptr$const$ptr
    }
  }
  // CHECK: }
}

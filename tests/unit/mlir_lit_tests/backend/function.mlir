//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --emit-c="tagless model=%S/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.defined<#clift.func<
  "/model-type/1001" : !void()>>

!f$ptr = !clift.ptr<8 to !f>

clift.module {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    unique_handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: fun_0x40001001_t *_var_0 = fun_0x40001001;
    clift.local !f$ptr "p" = {
      %f = clift.use @f : !f
      %r = clift.cast<decay> %f : !f -> !f$ptr
      clift.yield %r : !f$ptr
    }
  }
  // CHECK: }
}

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe import-model-names %S/model.yml %s /dev/stdout | %revngcliftopt | FileCheck %s

!void = !clift.primitive<void 0>
!uint64_t = !clift.primitive<unsigned 8>

!stack_1002_ = !clift.struct<"/type-definition/2004-StructDefinition" : size(8) {
  "/struct-field/2004-StructDefinition/0" : offset(0) !uint64_t
}>

// CHECK: !h = !clift.func<"/type-definition/1002-RawFunctionDefinition" as "h" :
// CHECK:   !void(!uint64_t, !stack_1002_)
// CHECK: >
!h = !clift.func<"/type-definition/1002-RawFunctionDefinition" : !void(!uint64_t, !stack_1002_)>

module attributes { clift.module } {
  // CHECK: clift.func @fun_0x40001002<!h>(
  // CHECK:   %arg0: !uint64_t {
  // CHECK:     clift.handle = "/raw-argument/1002-RawFunctionDefinition/rcx_x86_64"
  // CHECK:     clift.name = "rcx"
  // CHECK:   }
  // CHECK:   %arg1: !stack_1002_ {
  // CHECK:     clift.handle = "/raw-stack-arguments/1002-RawFunctionDefinition"
  // CHECK:     clift.name = "stack_arguments"
  // CHECK:   }
  // CHECK: ) -> !void
  // CHECK: attributes {
  // CHECK:   handle = "/function/0x40001002:Code_x86_64"
  // CHECK: }
  clift.func @h<!h>(%arg0 : !uint64_t { clift.handle = "/raw-argument/1002-RawFunctionDefinition/rcx_x86_64" },
                    %arg1 : !stack_1002_ { clift.handle = "/raw-stack-arguments/1002-RawFunctionDefinition" }) attributes {
    handle = "/function/0x40001002:Code_x86_64"
  } {
    clift.expr { clift.yield %arg0 : !uint64_t }
  }
}

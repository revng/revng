//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-expressions | FileCheck %s

!void = !clift.primitive<void 0>

!int32_t = !clift.primitive<signed 4>
!int64_t = !clift.primitive<signed 8>

!f = !clift.func<"/model-type/1001" : !void()>

module attributes {clift.module} {
  clift.func @f<!f>() -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.imm 3567587328 : !int32_t
      %0 = clift.imm 1000000000000 : !int64_t
      %1 = clift.cast<truncate> %0 : !int64_t -> !int32_t
      // CHECK: clift.yield %0 : !int32_t
      clift.yield %1 : !int32_t
    }
    // CHECK: }
  }
}

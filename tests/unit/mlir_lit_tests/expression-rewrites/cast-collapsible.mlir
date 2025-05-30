//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-expressions | FileCheck %s

!void = !clift.primitive<void 0>

!int16_t = !clift.primitive<signed 2>
!int32_t = !clift.primitive<signed 4>
!int64_t = !clift.primitive<signed 8>

!f = !clift.func<"/model-type/1001" : !void(!int64_t)>

module attributes {clift.module} {
  clift.func @f<!f>(%arg0 : !int64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.cast<truncate> %arg0 : !int64_t -> !int16_t
      %0 = clift.cast<truncate> %arg0 : !int64_t -> !int32_t
      %1 = clift.cast<truncate> %0 : !int32_t -> !int16_t
      // CHECK: clift.yield %0
      clift.yield %1 : !int16_t
    }
    // CHECK: }
  }
}

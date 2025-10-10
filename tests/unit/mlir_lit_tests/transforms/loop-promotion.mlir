//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --loop-promotion | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"f" : !void (!int32_t)>

!default_prototype = !clift.func<"default_prototype" : !void (!int32_t)>

module attributes {clift.module} {

  // CHECK: module attributes

  // Simple loop;

  clift.func @f<!default_prototype>(%arg0 : !int32_t) {

    %label_0 = clift.make_label

    clift.assign_label %label_0

    clift.if {
      clift.yield %arg0 : !int32_t
    } {
      clift.goto %label_0
    }
  }

  // CHECK-LABEL: clift.func @f

  // We check that the unique retreating edge is actually promoted to a
  // `clift.while` statement

  // CHECK: clift.while

  // We check that no other `clift.while` is emitted

  // CHECK-NOT: clift.while

  // Well nested loops;
  clift.func @g<!default_prototype>(%arg0 : !int32_t) {

    %label_0 = clift.make_label
    %label_1 = clift.make_label

    clift.assign_label %label_0

    clift.if {
      clift.yield %arg0 : !int32_t
    } {

      clift.assign_label %label_1
      clift.if {
        clift.yield %arg0 : !int32_t
      } {

        clift.goto %label_1
      }

      clift.goto %label_0
    }
  }

  // CHECK-LABEL: clift.func @g

  // We check that the two retreating edges present are both promoted to
  // `clift.while` statements

  // CHECK: clift.while
  // CHECK: clift.while

  // We check that no other `clift.while` is emitted

  // CHECK-NOT: clift.while

  // Overlapping loops;

  clift.func @h<!default_prototype>(%arg0 : !int32_t) {
    %label_0 = clift.make_label
    %label_1 = clift.make_label

    clift.assign_label %label_0

    clift.if {
      clift.yield %arg0 : !int32_t
    } {

      clift.assign_label %label_1
      clift.if {
        clift.yield %arg0 : !int32_t
      } {
        clift.goto %label_0
      }

      clift.goto %label_1
    }
  }

  // CHECK-LABEL: clift.func @h

  // We check that the two overlapping loops are promoted to `clift.while`
  // statements

  // CHECK: clift.while
  // CHECK: clift.while

  // We check that no other `clift.while` is emitted

  // CHECK-NOT: clift.while

  // Consecutive non-nested loops;
  clift.func @i<!default_prototype>(%arg0 : !int32_t) {
    %label_0 = clift.make_label
    %label_1 = clift.make_label

    clift.assign_label %label_0

    clift.if {
      clift.yield %arg0 : !int32_t
    } {
      clift.goto %label_0
    }

    clift.assign_label %label_1

    clift.if {
      clift.yield %arg0 : !int32_t
    } {
      clift.goto %label_1
    }
  }

  // CHECK-LABEL: clift.func @i

  // We check that the two consecutive non overlapping loops are promoted to
  // `clift.while` statements

  // CHECK: clift.while
  // CHECK: clift.while

  // We check that no other `clift.while` is emitted

  // CHECK-NOT: clift.while

}

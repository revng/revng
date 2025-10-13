//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --canonicalize | FileCheck %s

// CHECK: module {
module {
  // CHECK: %0 = clift.make_label
  %0 = clift.make_label
  clift.assign_label %0
  // CHECK: clift.assign_label %0
  clift.goto %0
  // CHECK: clift.goto %0
// CHECK: }
}

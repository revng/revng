//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s

// CHECK: #data_model = #clift.data_model<
#data_model = #clift.data_model<
  // CHECK-NEXT: pointer = 8
  pointer = 8,
  // CHECK-NOT: char
  char = 1,
  // CHECK-NOT: short
  short = 2,
  // CHECK-NOT: int
  int = 4,
  // CHECK-NEXT: long = 4
  long = 4,
  // CHECK-NOT: long long
  long long = 8,
  // CHECK-NOT: float
  float = 4,
  // CHECK-NOT: double
  double = 8,
  // CHECK-NOT: long double
  long double = 8
>
// CHECK-NEXT: >

module attributes { clift.data_model = #data_model } {}

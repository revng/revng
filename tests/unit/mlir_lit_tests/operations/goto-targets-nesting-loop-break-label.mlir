//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

%break = clift.make_label

clift.for break %break body {
  // CHECK: clift.goto may not target a nesting loop label
  clift.goto %break
}

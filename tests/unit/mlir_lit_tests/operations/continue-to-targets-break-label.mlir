//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

%break = clift.make_label

clift.for break %break body {
  // CHECK: clift.continue_to must target a loop continue label
  clift.continue_to %break
}

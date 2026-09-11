//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s

module attributes {
  // CHECK: clift.test_1 = #clift.switch<>,
  clift.test_1 = #clift.switch<>,

  // CHECK: clift.test_2 = #clift.switch<0, 1, 2>,
  clift.test_2 = #clift.switch<0, 1, 2>,

  // CHECK: clift.test_3 = #clift.switch<0, 1, 2>,
  clift.test_3 = #clift.switch<(0), (1), (2)>,

  // CHECK: clift.test_4 = #clift.switch<0, (1, 2), 3, (4, 5, 6)>
  clift.test_4 = #clift.switch<0, (1, 2), 3, (4, 5, 6)>
} {}

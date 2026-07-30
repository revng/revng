//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s | FileCheck %s

// An operand-less break_to/continue_to (a plain break/continue) is valid when
// nested directly within a loop.

clift.for body {
  // CHECK: clift.break_to{{$}}
  clift.break_to
}

clift.for body {
  // CHECK: clift.continue_to{{$}}
  clift.continue_to
}

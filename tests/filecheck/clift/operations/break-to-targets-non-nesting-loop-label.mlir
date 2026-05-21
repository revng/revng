//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

%break = clift.make_label

clift.for break %break body {}

// CHECK: clift.break_to must target a nesting loop label
clift.break_to %break

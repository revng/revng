//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

%label = clift.make_label
clift.assign_label %label

// CHECK: clift.break_to must target a loop break label
clift.break_to %label

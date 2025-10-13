//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

%label = clift.make_label
clift.assign_label %label

// CHECK: clift.continue_to must target a loop continue label
clift.continue_to %label

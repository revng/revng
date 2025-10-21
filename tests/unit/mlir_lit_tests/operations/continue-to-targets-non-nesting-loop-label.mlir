//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

%continue = clift.make_label

clift.for continue %continue body {}

// CHECK: clift.continue_to must target a nesting loop label
clift.continue_to %continue

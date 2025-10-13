//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

// CHECK: number of operation operands must equal the number of set label_mask flags
"clift.for"() ({}, {}, {}, {}) {label_mask = 1 : ui2} : () -> ()

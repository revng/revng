//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// CHECK: value is negative.
"clift.fimm"() {value = -0.0 : f32} : () -> !clift.float<4>

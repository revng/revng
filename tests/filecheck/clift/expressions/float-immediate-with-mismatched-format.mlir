//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// CHECK: value format does not match the operation type.
"clift.fimm"() {value = 1.0 : f64} : () -> !clift.float<4>

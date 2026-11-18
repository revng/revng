//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// CHECK: result #0 must be non-const floating point type
clift.fimm 1.0 : !clift.int<signed 4>

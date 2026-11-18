//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// CHECK: Expected non-negative real number.
clift.fimm -1.0 : !clift.float<4>

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// CHECK: Expected decimal real number.
clift.fimm 0x3F800000 : !clift.float<4>

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// CHECK: result element type must be effectively const
clift.str "hello" : !clift.array<6 x !clift.int<signed 1>>

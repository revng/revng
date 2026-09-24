//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// CHECK: result type length must match string length
clift.str "hello" : !clift.const<!clift.array<5 x !clift.int<signed 1>>>

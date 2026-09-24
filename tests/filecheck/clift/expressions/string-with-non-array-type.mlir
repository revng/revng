//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// CHECK: result must have array type
clift.str "hello" : !clift.const<!clift.int<signed 1>>

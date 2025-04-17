//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

// CHECK: non-Clift type
module attributes {clift.module, clift.test = i32} {}

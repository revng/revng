//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

// CHECK: failed to verify constraint: Region representing an expression
clift.do_while body {} cond {}

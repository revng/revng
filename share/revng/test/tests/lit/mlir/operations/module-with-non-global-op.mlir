//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

module attributes {clift.module} {
  // CHECK: clift.undef cannot be directly nested within a ModuleOp
  clift.undef : !clift.primitive<void 0>
}

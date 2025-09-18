//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe model-verify-clift %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>

module attributes {clift.module} {
  // CHECK: global variable with invalid handle: '/made-up-kind/something'
  clift.global @g : !int32_t attributes {
    handle = "/made-up-kind/something"
  }
}

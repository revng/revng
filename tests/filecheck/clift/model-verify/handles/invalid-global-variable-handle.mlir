//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>

module attributes {clift.module} {
  // CHECK: a global variable with an invalid handle: '/made-up-kind/something'
  clift.global @g : !int32_t attributes {
    handle = "/made-up-kind/something"
  }
}

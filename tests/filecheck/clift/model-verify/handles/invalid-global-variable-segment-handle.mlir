//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!int32_t = !clift.int<signed 4>

module attributes {clift.module} {
  // CHECK: a segment with an invalid handle: '/segment/0x40005000:Generic64'
  clift.global @g : !int32_t attributes {
    handle = "/segment/0x40005000:Generic64"
  }
}

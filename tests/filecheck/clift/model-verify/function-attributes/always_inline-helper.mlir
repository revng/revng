//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!void = !clift.void
!generic64_t = !clift.int<generic 8>

!revng_undefined_local_sp = !clift.func<"/helper-function/revng_undefined_local_sp" as "revng_undefined_local_sp" : !generic64_t()>

module attributes {clift.module} {

  // CHECK: `_ALWAYS_INLINE` is attached to a function that does not support attributes. See '/helper-function/revng_undefined_local_sp'

  clift.func @revng_undefined_local_sp<!revng_undefined_local_sp>() -> !generic64_t attributes {
    handle = "/helper-function/revng_undefined_local_sp",
    always_inline
  }

}

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng2 pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

!f_2 = !clift.func<
  "/type-definition/2-RawFunctionDefinition" : !uint64_t(!uint64_t)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>
  ]
>

// CHECK: A non c-attribute was found among the `c_attributes` in '/function/0x1004:Code_aarch64'

module attributes { clift.module } {

  clift.func @f_2<!f_2>(
    !uint64_t {
      clift.c_attributes = [!uint64_t],
      clift.handle = "/cabi-argument/2-RawFunctionDefinition/x0_aarch64"
    }
  ) -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x1004:Code_aarch64"
  }

}

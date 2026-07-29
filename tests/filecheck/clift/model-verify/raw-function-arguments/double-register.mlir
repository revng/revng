//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng2 pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

!s_0 = !clift.struct<"/type-definition/0-StructDefinition" : size(64) {}>

!f_1 = !clift.func<
  "/type-definition/1-RawFunctionDefinition" : !void(!uint64_t)
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>]
>

module attributes {clift.module} {

  // CHECK: More than one `_REG` attribute is attached to '/raw-stack-arguments/1-RawFunctionDefinition' of '/function/0x1004:Code_aarch64'

  clift.func @f_1<!f_1>(
    !uint64_t {
      clift.c_attributes = [
        #clift.c_attribute<"_REG" : "/macro/_REG" [#clift.identifier<"x0_aarch64">]>,
        #clift.c_attribute<"_REG" : "/macro/_REG" [#clift.identifier<"x1_aarch64">]>
      ],
      clift.handle = "/raw-stack-arguments/1-RawFunctionDefinition"
    }
  ) -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x1004:Code_aarch64"
  }

}

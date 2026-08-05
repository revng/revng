//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not FileCheck < <(%root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null -- --debug-log=model-verify 2>&1)

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

!f_3 = !clift.func<
  "/type-definition/3-CABIFunctionDefinition" : !void(!uint64_t)
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"AAPCS64">]>]
>

module attributes {clift.module} {

  // CHECK: `_REG` attribute is only allowed on functions with a *raw* prototype. See '/cabi-argument/3-CABIFunctionDefinition/0' of '/function/0x10d4:Code_aarch64'

  clift.func @f_3<!f_3>(
    !uint64_t {
      clift.c_attributes = [#clift.c_attribute<"_REG" : "/macro/_REG" [#clift.identifier<"x0_aarch64">]>],
      clift.handle = "/cabi-argument/3-CABIFunctionDefinition/0"
    }
  ) -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x10d4:Code_aarch64"
  }

}

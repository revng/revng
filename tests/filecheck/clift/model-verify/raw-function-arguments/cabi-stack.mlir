//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

!f_3 = !clift.func<
  "/type-definition/3-CABIFunctionDefinition" : !void(!uint64_t)
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"AAPCS64">]>]
>

module attributes {clift.module} {

  // CHECK: `_STACK` attribute is only allowed on functions with a *raw* prototype. See '/cabi-argument/3-CABIFunctionDefinition/0' of '/function/0x10d4:Code_aarch64'

  clift.func @f_3<!f_3>(
    !uint64_t {
      clift.c_attributes = [#clift.c_attribute<"_STACK" : "/macro/_STACK">],
      clift.handle = "/cabi-argument/3-CABIFunctionDefinition/0"
    }
  ) -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x10d4:Code_aarch64"
  }

}

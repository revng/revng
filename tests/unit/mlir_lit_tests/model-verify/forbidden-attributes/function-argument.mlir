//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

!f_2 = !clift.func<
  "/type-definition/2-RawFunctionDefinition" : !uint64_t(!uint64_t)
  [
    #clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>
  ]
>

// CHECK: Forbidden c-attribute ('_ABI') found in '/cabi-argument/2-RawFunctionDefinition/x0_aarch64' of '/function/0x1004:Code_aarch64'

module attributes { clift.module } {

  clift.func @f_2<!f_2>(
    !uint64_t {
      clift.c_attributes = [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>],
      clift.handle = "/cabi-argument/2-RawFunctionDefinition/x0_aarch64"
    }
  ) -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x1004:Code_aarch64"
  }

}

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng pipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

!s_0 = !clift.struct<"/type-definition/0-StructDefinition" : size(64) {}>

!f_1 = !clift.func<
  "/type-definition/1-RawFunctionDefinition" : !void(!s_0)
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>]
>

module attributes {clift.module} {

  // CHECK: More than one `_STACK` attribute is attached to '/raw-stack-arguments/1-RawFunctionDefinition' of '/function/0x1004:Code_aarch64'

  clift.func @f_1<!f_1>(
    !s_0 {
      clift.c_attributes = [
        #clift.c_attribute<"_STACK" : "/macro/_STACK">,
        #clift.c_attribute<"_STACK" : "/macro/_STACK">
      ],
      clift.handle = "/raw-stack-arguments/1-RawFunctionDefinition"
    }
  ) -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x1004:Code_aarch64"
  }

}

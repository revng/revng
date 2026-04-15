//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngpipe verify-against-model %S/model.yml %s /dev/null 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>

!f = !clift.func<
  "/type-definition/0-CABIFunctionDefinition" : !void()
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"AAPCS64">]>]
>

module attributes {clift.module} {

  // CHECK: Attached function attribute count ('0') does not match the model value ('1'). See '/function/0x1024:Code_aarch64'

  clift.func @f_2<!f>() -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x1024:Code_aarch64"
  }

}

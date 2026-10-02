//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not --crash %root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(%pipe_input %s) /dev/null -- --debug-log=model-verify 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>

!f = !clift.func<
  "/type-definition/0-CABIFunctionDefinition" : !void()
  #clift.c_attribute_list<
    <"_ABI" : "/macro/_ABI" [#clift.identifier<"AAPCS64">]>
  >
>

module attributes {clift.module} {

  // CHECK: `_NO_RETURN` is attached to a function that does not have it in the model. See '/function/0x1004:Code_aarch64'

  clift.func @f_1<!f>() -> !void attributes {
    handle = "/function/0x1004:Code_aarch64",
    noreturn
  }

}

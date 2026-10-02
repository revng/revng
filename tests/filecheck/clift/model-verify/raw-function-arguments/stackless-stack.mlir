//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not --crash %root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(%pipe_input %s) /dev/null -- --debug-log=model-verify 2>&1 | FileCheck %s

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

!f_2 = !clift.func<
  "/type-definition/2-RawFunctionDefinition" : !uint64_t(!uint64_t)
  #clift.c_attribute_list<
    <"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>
  >
>

module attributes {clift.module} {

  // CHECK: `_STACK` attribute is only allowed on functions that have stack arguments in the model. See '/raw-argument/2-RawFunctionDefinition/x3_aarch64' of '/function/0x1064:Code_aarch64'

  clift.func @f_2<!f_2>(
    !uint64_t {
      clift.c_attribute_list = #clift.c_attribute_list<
        <"_STACK" : "/macro/_STACK">
      >,
      clift.handle = "/raw-argument/2-RawFunctionDefinition/x3_aarch64"
    }
  ) -> !uint64_t attributes {
    handle = "/function/0x1064:Code_aarch64"
  }

}

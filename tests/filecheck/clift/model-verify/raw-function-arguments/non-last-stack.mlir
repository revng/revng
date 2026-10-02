//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not --crash %root/bin/revng pipeline run-pipe verify-against-model %S/model.yml <(%pipe_input %s) /dev/null -- --debug-log=model-verify 2>&1 | FileCheck %s

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

!s_0 = !clift.struct<"/type-definition/0-StructDefinition" : size(64) {}>

!f_1 = !clift.func<
  "/type-definition/1-RawFunctionDefinition" : !void(!s_0, !uint64_t)
  #clift.c_attribute_list<
    <"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>
  >
>

module attributes {clift.module} {

  // CHECK: `_STACK` attribute must not have any arguments. See '/raw-stack-arguments/1-RawFunctionDefinition' of '/function/0x1004:Code_aarch64'

  clift.func @f_1<!f_1>(
    !s_0 {
      clift.c_attribute_list = #clift.c_attribute_list<
        <"_STACK" : "/macro/_STACK" [42]>
      >,
      clift.handle = "/raw-stack-arguments/1-RawFunctionDefinition"
    },
    !uint64_t {
      clift.c_attribute_list = #clift.c_attribute_list<
        <"_REG" : "/macro/_REG" [#clift.identifier<"x0_aarch64">]>
      >,
      clift.handle = "/raw-argument/1-RawFunctionDefinition/x0_aarch64"
    }
  ) -> !void attributes {
    handle = "/function/0x1004:Code_aarch64"
  }

}

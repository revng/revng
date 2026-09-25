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

// CHECK: Forbidden c-attribute ('_ABI') found in '/cabi-argument/2-RawFunctionDefinition/x0_aarch64' of '/function/0x1004:Code_aarch64'

module attributes { clift.module } {

  clift.func @f_2<!f_2>(
    !uint64_t {
      clift.c_attribute_list = #clift.c_attribute_list<<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>>,
      clift.handle = "/cabi-argument/2-RawFunctionDefinition/x0_aarch64"
    }
  ) -> !void attributes {
    handle = "/function/0x1004:Code_aarch64"
  }

}

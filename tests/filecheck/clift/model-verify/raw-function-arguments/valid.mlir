//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng2 pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null

!void = !clift.void
!uint64_t = !clift.int<unsigned 8>

!s_0 = !clift.struct<"/type-definition/0-StructDefinition" : size(64) {}>

!rv_1 = !clift.struct<"/artificial-struct/1-RawFunctionDefinition" : size(16) {
  "/return-register/1-RawFunctionDefinition/x0_aarch64" : offset(0) !uint64_t,
  "/return-register/1-RawFunctionDefinition/x1_aarch64" : offset(8) !uint64_t
}>
!f_1 = !clift.func<
  "/type-definition/1-RawFunctionDefinition" : !rv_1(!uint64_t, !uint64_t, !s_0)
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>]
>

!f_2 = !clift.func<
  "/type-definition/2-RawFunctionDefinition" : !uint64_t(!uint64_t)
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"raw_aarch64">]>]
>

!f_3 = !clift.func<
  "/type-definition/3-CABIFunctionDefinition" : !void(!uint64_t)
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"AAPCS64">]>]
>

module attributes {clift.module} {

  clift.func @f_1<!f_1>(
    !uint64_t {
      clift.c_attributes = [#clift.c_attribute<"_REG" : "/macro/_REG" [#clift.identifier<"x0_aarch64">]>],
      clift.handle = "/raw-argument/1-RawFunctionDefinition/x0_aarch64"
    },
    !uint64_t {
      clift.c_attributes = [#clift.c_attribute<"_REG" : "/macro/_REG" [#clift.identifier<"x1_aarch64">]>],
      clift.handle = "/raw-argument/1-RawFunctionDefinition/x1_aarch64"
    },
    !s_0 {
      clift.c_attributes = [#clift.c_attribute<"_STACK" : "/macro/_STACK">],
      clift.handle = "/raw-stack-arguments/1-RawFunctionDefinition"
    }
  ) -> !rv_1 attributes {
    clift.c_attributes = [],
    handle = "/function/0x1004:Code_aarch64"
  }

  clift.func @f_2<!f_2>(
    !uint64_t {
      clift.c_attributes = [#clift.c_attribute<"_REG" : "/macro/_REG" [#clift.identifier<"x3_aarch64">]>],
      clift.handle = "/raw-argument/2-RawFunctionDefinition/x3_aarch64"
    }
  ) -> !uint64_t attributes {
    clift.c_attributes = [],
    handle = "/function/0x1064:Code_aarch64"
  }

  clift.func @f_3<!f_3>(
    !uint64_t {
      clift.c_attributes = [],
      clift.handle = "/cabi-argument/3-CABIFunctionDefinition/0"
    }
  ) -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x10d4:Code_aarch64"
  }

}

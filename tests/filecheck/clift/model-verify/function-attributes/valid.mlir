//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng2 pipeline run-pipe verify-against-model %S/model.yml <(tar -c --transform 's;.*;/binary;' %s) /dev/null

!void = !clift.void
!generic64_t = !clift.int<generic 8>

!f = !clift.func<
  "/type-definition/0-CABIFunctionDefinition" : !void()
  [#clift.c_attribute<"_ABI" : "/macro/_ABI" [#clift.identifier<"AAPCS64">]>]
>

!revng_undefined_local_sp = !clift.func<"/helper-function/revng_undefined_local_sp" as "revng_undefined_local_sp" : !generic64_t()>

module attributes {clift.module} {

  clift.func @f_1<!f>() -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x1004:Code_aarch64"
  }
  clift.func @f_2<!f>() -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x1024:Code_aarch64",
    noreturn
  }
  clift.func @f_3<!f>() -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x1044:Code_aarch64",
    always_inline
  }
  clift.func @f_4<!f>() -> !void attributes {
    clift.c_attributes = [],
    handle = "/function/0x1064:Code_aarch64",
    noreturn,
    always_inline
  }

  clift.func @revng_undefined_local_sp<!revng_undefined_local_sp>() -> !generic64_t attributes {
    handle = "/helper-function/revng_undefined_local_sp"
  }

}

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>

!f = !clift.func<"f_t" as "f" : !void()>
!s = !clift.struct<"s" as "s" : size(1) {}>

module attributes {clift.module} {
  // CHECK: void f(void) {
  clift.func @f<!f>() attributes {
    clift.handle = "f",
    clift.name = "f"
  } {
    // CHECK: const s *const cp8_cs;
    clift.local : !clift.const<!clift.ptr<8 to !clift.const<!s>>> attributes {
      clift.handle = "cp8_cs",
      clift.name = "cp8_cs"
    }

    // CHECK: const pointer32_t(const s) cp4_cs;
    clift.local : !clift.const<!clift.ptr<4 to !clift.const<!s>>> attributes {
      clift.handle = "cp4_cs",
      clift.name = "cp4_cs"
    }

    // CHECK: const s *const *const cp8_cp8_cs;
    clift.local : !clift.const<!clift.ptr<8 to !clift.const<!clift.ptr<8 to !clift.const<!s>>>>> attributes {
      clift.handle = "cp8_cp8_cs",
      clift.name = "cp8_cp8_cs"
    }

    // CHECK: const pointer32_t(const pointer32_t(const s)) cp4_cp4_cs;
    clift.local : !clift.const<!clift.ptr<4 to !clift.const<!clift.ptr<4 to !clift.const<!s>>>>> attributes {
      clift.handle = "cp4_cp4_cs",
      clift.name = "cp4_cp4_cs"
    }

    // CHECK: const s a1_cs[1];
    clift.local : !clift.array<1 x !clift.const<!s>> attributes {
      clift.handle = "a1_cs",
      clift.name = "a1_cs"
    }

    // CHECK: const s a2_a1_cs[2][1];
    clift.local : !clift.array<2 x !clift.array<1 x !clift.const<!s>>> attributes {
      clift.handle = "a2_a1_cs",
      clift.name = "a2_a1_cs"
    }

    // CHECK: const s(*const cp8_a1_cs)[1];
    clift.local : !clift.const<!clift.ptr<8 to !clift.array<1 x !clift.const<!s>>>> attributes {
      clift.handle = "cp8_a1_cs",
      clift.name = "cp8_a1_cs"
    }

    // CHECK: const pointer32_t(const s[1]) cp4_a1_cs;
    clift.local : !clift.const<!clift.ptr<4 to !clift.array<1 x !clift.const<!s>>>> attributes {
      clift.handle = "cp4_a1_cs",
      clift.name = "cp4_a1_cs"
    }

    // CHECK: const s *const a1_cp8_cs[1];
    clift.local : !clift.array<1 x !clift.const<!clift.ptr<8 to !clift.const<!s>>>> attributes {
      clift.handle = "a1_cp8_cs",
      clift.name = "a1_cp8_cs"
    }

    // CHECK: const pointer32_t(const s) a1_cp4_cs[1];
    clift.local : !clift.array<1 x !clift.const<!clift.ptr<4 to !clift.const<!s>>>> attributes {
      clift.handle = "a1_cp4_cs",
      clift.name = "a1_cp4_cs"
    }

    // CHECK: const s(*const a2_cp8_a1_cs[2])[1];
    clift.local : !clift.array<2 x !clift.const<!clift.ptr<8 to !clift.array<1 x !clift.const<!s>>>>> attributes {
      clift.handle = "a2_cp8_a1_cs",
      clift.name = "a2_cp8_a1_cs"
    }

    // CHECK: const pointer32_t(const s[1]) a2_cp4_a1_cs[2];
    clift.local : !clift.array<2 x !clift.const<!clift.ptr<4 to !clift.array<1 x !clift.const<!s>>>>> attributes {
      clift.handle = "a2_cp4_a1_cs",
      clift.name = "a2_cp4_a1_cs"
    }
  }
  // CHECK: }
}

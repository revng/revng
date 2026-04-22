//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// NOTE: this directory is not a perfect fit for this test, but it's the only
//       non-comment-related header test, so I'm putting it here instead of
//       making a subdirectory of its own.

// RUN: %revngcliftopt --emit-helper-header %s -o /dev/null | FileCheck %s
// RUN: %revngcliftopt --emit-helper-header=ptml %s -o /dev/null | %revngptml | FileCheck %s

!void = !clift.void
!generic32_t = !clift.int<generic 4>
!generic64_t = !clift.int<generic 8>

!lshift = !clift.func<"/helper-function/lshift" as "lshift" : !generic64_t(!generic64_t, !generic32_t)>
!revng_undefined_local_sp = !clift.func<"/helper-function/revng_undefined_local_sp" as "revng_undefined_local_sp" : !generic64_t()>
!helper_syscall_wrapper = !clift.func<"/helper-function/helper_syscall_wrapper" as "helper_syscall_wrapper" : !void(!clift.ptr<8 to !void>, !generic32_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>)>

module attributes {clift.module} {
  // CHECK: generic64_t lshift(generic64_t, generic32_t);
  clift.func @lshift<!lshift>(!generic64_t, !generic32_t) -> !generic64_t attributes {handle = "/helper-function/lshift"}

  // CHECK: generic64_t revng_undefined_local_sp(void);
  clift.func @revng_undefined_local_sp<!revng_undefined_local_sp>() -> !generic64_t attributes {handle = "/helper-function/revng_undefined_local_sp"}

  // CHECK: void helper_syscall_wrapper(void *, generic32_t, generic64_t, generic64_t, generic64_t, generic64_t, generic64_t, generic64_t, generic64_t, generic64_t, generic64_t, generic64_t, generic64_t, generic64_t, generic64_t, void *, void *, void *, void *, void *, void *, void *);
  clift.func @helper_syscall_wrapper<!helper_syscall_wrapper>(!clift.ptr<8 to !void>, !generic32_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !generic64_t, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>, !clift.ptr<8 to !void>) -> !void attributes {handle = "/helper-function/helper_syscall_wrapper"}
}

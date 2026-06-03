;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt -mark-inline-helpers-up-to %s -S -o - | FileCheck %s

; The pass tags every helper transitively reaching a function defined under
; one of the configured runtime-library directories (currently `fpu/`) with
; the `revng_inline` section. The leaf functions themselves and unrelated
; helpers must NOT be tagged.

; A function defined under `fpu/`: it is the runtime-library leaf. NOT tagged.
; CHECK:      define i64 @softfloat_leaf{{[^\n]*}}{{$}}
; CHECK-NOT:  section "revng_inline"
define i64 @softfloat_leaf(i64 %a) !dbg !10 {
  ret i64 %a
}

; A direct caller of the leaf: tagged.
; CHECK:      define i64 @helper_direct{{[^\n]*}}section "revng_inline"
define i64 @helper_direct(i64 %a) {
  %r = call i64 @softfloat_leaf(i64 %a)
  ret i64 %r
}

; A transitive caller (calls `helper_direct` which calls the leaf): tagged.
; CHECK:      define i64 @helper_transitive{{[^\n]*}}section "revng_inline"
define i64 @helper_transitive(i64 %a) {
  %r = call i64 @helper_direct(i64 %a)
  ret i64 %r
}

; A helper that does NOT reach the leaf: never tagged.
; CHECK:      define i64 @helper_unrelated{{[^\n]*}}{{$}}
; CHECK-NOT:  section "revng_inline"
define i64 @helper_unrelated(i64 %a) {
  %r = add i64 %a, 42
  ret i64 %r
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2)
!2 = !DIFile(filename: "fpu/softfloat.c", directory: "")
!10 = distinct !DISubprogram(file: !2, unit: !1)

;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; `inlined_callee` is marked as `AlwaysInline`, so its body is emitted next to
; `caller` and inlined at both of the call sites, which disappear.

CHECK-LABEL: define void @local_caller()
CHECK-NOT:     call void @local_inlined_callee()

; A call to a function that is not marked is left alone.

CHECK-LABEL: define void @local_other_caller()
CHECK:         call void @local_caller()

CHECK-LABEL: define void @local_main()
CHECK:         call void @local_other_caller()

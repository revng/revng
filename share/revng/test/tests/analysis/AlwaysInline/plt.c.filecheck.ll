;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; `detect-abi` marks every PLT entry as `AlwaysInline`, so no caller reaches a
; dynamic function through its stub anymore.

CHECK-LABEL: define void @local_use_plt()
CHECK-NOT:     call void @local_malloc_2()
CHECK-NOT:     call void @local_free_2()
CHECK-DAG:     call void @dynamic_malloc()
CHECK-DAG:     call void @dynamic_free()

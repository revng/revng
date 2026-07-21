;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt %s -extractvalue-to-gep -S -o - | FileCheck %s

; =============================================================================
; =============================================================================
; Tests that extractvalue is turned into a load from an i8 GEP into a copy of
; the aggregate, and crucially that the GEP is only emitted when the accessed
; field is at a non-zero offset. A field at offset 0 must be loaded directly
; from the base pointer, with no `getelementptr ..., i64 0`.
; =============================================================================
; =============================================================================

declare { i64, i16 } @make_struct()

; A struct value produced by a call is spilled to an alloca, then its fields are
; read back. Field 0 (offset 0) must be a direct load from the alloca; field 1
; (offset 8) must go through a GEP with offset 8. No zero-offset GEP may appear.
;
; CHECK-LABEL: define void @struct_from_call
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca { i64, i16 }
; CHECK: [[STRUCT:%[a-zA-Z0-9_]+]] = call { i64, i16 } @make_struct()
; CHECK: store { i64, i16 } [[STRUCT]], ptr [[ALLOCA]]
; CHECK-NOT: getelementptr i8, ptr [[ALLOCA]], i64 0
; CHECK: [[FIELD0:%[a-zA-Z0-9_]+]] = load i64, ptr [[ALLOCA]]
; CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[ALLOCA]], i64 8
; CHECK: [[FIELD1:%[a-zA-Z0-9_]+]] = load i16, ptr [[GEP1]]
; CHECK: store i64 [[FIELD0]], ptr null
; CHECK: store i16 [[FIELD1]], ptr null
define void @struct_from_call() {
  %s = call { i64, i16 } @make_struct()
  %f0 = extractvalue { i64, i16 } %s, 0
  %f1 = extractvalue { i64, i16 } %s, 1
  store i64 %f0, ptr null
  store i16 %f1, ptr null
  ret void
}

; A struct value that is already in memory (the aggregate operand is a load)
; reuses the load's pointer as the base, so no extra alloca is created. Field 0
; (offset 0) is a direct load from the base; field 1 (offset 4) goes through a
; GEP with offset 4. Again no zero-offset GEP may appear.
;
; CHECK-LABEL: define void @struct_from_load
; CHECK-NOT: getelementptr i8, ptr %p, i64 0
; CHECK: [[LF0:%[a-zA-Z0-9_]+]] = load i32, ptr %p
; CHECK: [[LGEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr %p, i64 4
; CHECK: [[LF1:%[a-zA-Z0-9_]+]] = load i32, ptr [[LGEP]]
; CHECK: store i32 [[LF0]], ptr null
; CHECK: store i32 [[LF1]], ptr null
define void @struct_from_load(ptr %p) {
  %s = load { i32, i32 }, ptr %p
  %f0 = extractvalue { i32, i32 } %s, 0
  %f1 = extractvalue { i32, i32 } %s, 1
  store i32 %f0, ptr null
  store i32 %f1, ptr null
  ret void
}

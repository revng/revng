;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt -enable-new-pm=true -load-pass-plugin=%root/lib/revng/analyses/librevngCanonicalize.so -passes=switch-to-statements-test %s -S -o - | FileCheck %s

@segment = external global i64

; =================================
; Arguments and locals don't alias.
; =================================

; Function argument doesn't alias the alloca.
; No new local variables should be created.
;
; CHECK-LABEL: define i64 @a
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NOT: alloca

define i64 @a(ptr %arg) {
  %local = alloca i64
  store i64 15, ptr %local
  %fifteen = load i64, ptr %local
  store i64 7, ptr %arg
  ret i64 %fifteen
}

; Function argument doesn't alias the alloca.
; No new local variables should be created.
;
; This is the same as the previous test, but with alloca and argument swapped.
;
; CHECK-LABEL: define i64 @a_swapped
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NOT: alloca

define i64 @a_swapped(ptr %arg) {
  %local = alloca i64
  store i64 15, ptr %arg
  %fifteen = load i64, ptr %arg
  store i64 7, ptr %local
  ret i64 %fifteen
}

; =================================
; Arguments and globals may alias.
; =================================

; Function argument may alias the @segment global.
; A new local variable should be created, and the result of the load should be
; stored in there.
; The the ret instruction should load from there again.
;
; CHECK-LABEL: define i64 @b
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NEXT: [[LOADED_VALUE:%[a-zA-Z0-9_]+]] = load i64, ptr %arg
; CHECK-NEXT: store i64 [[LOADED_VALUE]], ptr [[ALLOCA]]
; CHECK-NEXT: store i64 7, ptr @segment
; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9_]+]] = load i64, ptr [[ALLOCA]]
; CHECK-NEXT: ret i64 [[RESULT]]

define i64 @b(ptr %arg) {
  %loaded_value = load i64, ptr %arg
  store i64 7, ptr @segment
  ret i64 %loaded_value
}

; Function argument may alias the @segment global.
; A new local variable should be created, and the result of the load should be
; stored in there.
; The the ret instruction should load from there again.
;
; This is the same as the previous test, but with segment and argument swapped.
;
; CHECK-LABEL: define i64 @b_swapped
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NEXT: [[LOADED_VALUE:%[a-zA-Z0-9_]+]] = load i64, ptr @segment
; CHECK-NEXT: store i64 [[LOADED_VALUE]], ptr [[ALLOCA]]
; CHECK-NEXT: store i64 7, ptr %arg
; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9_]+]] = load i64, ptr [[ALLOCA]]
; CHECK-NEXT: ret i64 [[RESULT]]

define i64 @b_swapped(ptr %arg) {
  %loaded_value = load i64, ptr @segment
  store i64 7, ptr %arg
  ret i64 %loaded_value
}

; ===============================
; Locals and globals don't alias.
; ===============================

; Segment global doesn't alias the alloca.
; No new local variables should be created.
;
; CHECK-LABEL: define i64 @c
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NOT: alloca

define i64 @c() {
  %local = alloca i64
  store i64 15, ptr %local
  %fifteen = load i64, ptr %local
  store i64 7, ptr @segment
  ret i64 %fifteen
}

; Segment global doesn't alias the alloca.
; No new local variables should be created.
;
; This is the same as the previous test, but with alloca and argument swapped.
;
; CHECK-LABEL: define i64 @c_swapped
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NOT: alloca

define i64 @c_swapped() {
  %local = alloca i64
  store i64 15, ptr @segment
  %fifteen = load i64, ptr @segment
  store i64 7, ptr %local
  ret i64 %fifteen
}

; ============================================================================
; Non-overlapping accesses to the same allocation (local, global, or argument)
; don't alias, so they don't force the emission of a new local variable.
; ============================================================================

; CHECK-LABEL: define i32 @d
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NOT: alloca
define i32 @d() {
  %stack = alloca i64
  %stack_at_0 = getelementptr i8, ptr %stack, i64 0
  %stack_at_4 = getelementptr i8, ptr %stack_at_0, i64 4
  store i32 15, ptr %stack_at_0
  %fifteen = load i32, ptr %stack_at_0
  store i32 7, ptr %stack_at_4
  ret i32 %fifteen
}

; CHECK-LABEL: define i32 @d_global
; CHECK-NOT: alloca
define i32 @d_global() {
  %segment_at_0 = getelementptr i8, ptr @segment, i64 0
  %segment_at_4 = getelementptr i8, ptr %segment_at_0, i64 4
  store i32 15, ptr %segment_at_4
  %fifteen = load i32, ptr %segment_at_4
  store i32 7, ptr %segment_at_0
  ret i32 %fifteen
}

; CHECK-LABEL: define i32 @d_argument
; CHECK-NOT: alloca
define i32 @d_argument(ptr %arg) {
  %arg_at_0 = getelementptr i8, ptr %arg, i64 0
  %arg_at_4 = getelementptr i8, ptr %arg_at_0, i64 4
  store i32 15, ptr %arg_at_0
  %fifteen = load i32, ptr %arg_at_0
  store i32 7, ptr %arg_at_4
  ret i32 %fifteen
}


; ============================================================================
; LLVM struct types are not special.
; ============================================================================

%s = type { i32, i32 }

; Function argument may alias the @segment global.
; A new local variable should be created, and the result of the load should be
; stored in there.
; The the ret instruction should load from there again.
;
; CHECK-LABEL: define %s @s
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca %s
; CHECK-NEXT: [[LOADED_VALUE:%[a-zA-Z0-9_]+]] = load %s, ptr %arg
; CHECK-NEXT: store %s [[LOADED_VALUE]], ptr [[ALLOCA]]
; CHECK-NEXT: store i64 7, ptr @segment
; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9_]+]] = load %s, ptr [[ALLOCA]]
; CHECK-NEXT: ret %s [[RESULT]]

define %s @s(ptr %arg) {
  %loaded_value = load %s, ptr %arg
  store i64 7, ptr @segment
  ret %s %loaded_value
}

; Function argument may alias the @segment global.
; A new local variable should be created, and the result of the load should be
; stored in there.
; The the ret instruction should load from there again.
;
; This is the same as the previous test, but with segment and argument swapped.
;
; CHECK-LABEL: define %s @s_swapped
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca %s
; CHECK-NEXT: [[LOADED_VALUE:%[a-zA-Z0-9_]+]] = load %s, ptr @segment
; CHECK-NEXT: store %s [[LOADED_VALUE]], ptr [[ALLOCA]]
; CHECK-NEXT: store i64 7, ptr %arg
; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9_]+]] = load %s, ptr [[ALLOCA]]
; CHECK-NEXT: ret %s [[RESULT]]

define %s @s_swapped(ptr %arg) {
  %loaded_value = load %s, ptr @segment
  store i64 7, ptr %arg
  ret %s %loaded_value
}

;; We should test that a call that returns a struct and an store of it into an
;; array-typed alloca don't create an additional struct typed local variable

; ============================================================================
; Calls handled correctly based on memory effects and number of uses.
; ============================================================================

declare i64 @rw_func(ptr) memory(readwrite) nounwind willreturn
declare i64 @ro_func(ptr) memory(read) nounwind willreturn

; A call to a read+write function whose only use is an icmp that is itself
; only used by a conditional branch. Because the call has a single use, it
; is not picked for serialisation: no new alloca is created.
;
; CHECK-LABEL: define i64 @call_rw_one_use(
; CHECK-NOT: alloca

define i64 @call_rw_one_use(ptr %arg) {
entry:
  %call = call i64 @rw_func(ptr %arg)
  %cmp = icmp ne i64 %call, 0
  br i1 %cmp, label %then, label %else

then:
  ret i64 1

else:
  ret i64 2
}

; Same shape as @call_rw_one_use, but the call has two uses (the icmp and a
; ret in the `then` block). Since the function may read+write memory and
; the call has more than one use, the call is picked for serialisation:
; - a new alloca of the call's return type is created at the top of entry,
; - the result of the call is stored into the alloca right after the call,
; - every user of the call uses a load from the alloca instead of the call
;   directly.
;
; CHECK-LABEL: define i64 @call_rw_two_uses(
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NEXT: [[CALL:%[a-zA-Z0-9_]+]] = call i64 @rw_func(ptr %arg)
; CHECK-NEXT: store i64 [[CALL]], ptr [[ALLOCA]]
; CHECK-NEXT: [[CMP_LD:%[a-zA-Z0-9_]+]] = load i64, ptr [[ALLOCA]]
; CHECK-NEXT: [[CMP:%[a-zA-Z0-9_]+]] = icmp ne i64 [[CMP_LD]], 0
; CHECK-NEXT: br i1 [[CMP]], label %then, label %else
; CHECK: then:
; CHECK-NEXT: [[RET_LD:%[a-zA-Z0-9_]+]] = load i64, ptr [[ALLOCA]]
; CHECK-NEXT: ret i64 [[RET_LD]]
; CHECK: else:
; CHECK-NEXT: ret i64 0

define i64 @call_rw_two_uses(ptr %arg) {
entry:
  %call = call i64 @rw_func(ptr %arg)
  %cmp = icmp ne i64 %call, 0
  br i1 %cmp, label %then, label %else

then:
  ret i64 %call

else:
  ret i64 0
}

; Same shape as @call_rw_two_uses, but the called function only reads
; memory. The call is not considered to have side effects, so it is not
; picked for serialisation: no new alloca is created.
;
; CHECK-LABEL: define i64 @call_ro_two_uses(
; CHECK-NOT: alloca

define i64 @call_ro_two_uses(ptr %arg) {
entry:
  %call = call i64 @ro_func(ptr %arg)
  %cmp = icmp ne i64 %call, 0
  br i1 %cmp, label %then, label %else

then:
  ret i64 %call

else:
  ret i64 0
}

; ============================================================================
; Same three call patterns as above, but with an extra i64 argument %val, a
; pre-existing alloca in the entry block, a store of %val into the alloca
; right after the call, and a load+ret from the alloca.
;
; The pre-existing alloca/store/load do not write to memory the call could
; read (the alloca is local) and the call does not write to the alloca, so
; they should not affect the picker's decision: each variant produces the
; same NEW alloca count as its no-extra-pieces counterpart above.
; ============================================================================

; Variant of @call_rw_one_use: still a single use of the call, no NEW
; alloca generated, only the pre-existing one survives.
;
; CHECK-LABEL: define i64 @call_rw_one_use_with_alloca(
; CHECK: alloca i64
; CHECK-NOT: alloca

define i64 @call_rw_one_use_with_alloca(ptr %arg, i64 %val) {
entry:
  %loc = alloca i64
  %call = call i64 @rw_func(ptr %arg)
  store i64 %val, ptr %loc
  %cmp = icmp ne i64 %call, 0
  br i1 %cmp, label %then, label %else

then:
  %loaded_then = load i64, ptr %loc
  ret i64 %loaded_then

else:
  %loaded_else = load i64, ptr %loc
  ret i64 %loaded_else
}

; Variant of @call_rw_two_uses: the call still has two uses (the icmp and
; the ret in `then`), so it is still picked. One NEW alloca is generated
; (in addition to the pre-existing one), the call's result is stored into
; the new alloca, and every user of the call uses a load from it.
;
; CHECK-LABEL: define i64 @call_rw_two_uses_with_alloca(
; CHECK: alloca i64
; CHECK: alloca i64
; CHECK-NOT: alloca

define i64 @call_rw_two_uses_with_alloca(ptr %arg, i64 %val) {
entry:
  %loc = alloca i64
  %call = call i64 @rw_func(ptr %arg)
  store i64 %val, ptr %loc
  %cmp = icmp ne i64 %call, 0
  br i1 %cmp, label %then, label %else

then:
  ret i64 %call

else:
  %loaded = load i64, ptr %loc
  ret i64 %loaded
}

; Variant of @call_ro_two_uses: the call is read-only, so it is still not
; picked for serialisation. No NEW alloca is generated; only the
; pre-existing one survives.
;
; CHECK-LABEL: define i64 @call_ro_two_uses_with_alloca(
; CHECK: alloca i64
; CHECK-NOT: alloca

define i64 @call_ro_two_uses_with_alloca(ptr %arg, i64 %val) {
entry:
  %loc = alloca i64
  %call = call i64 @ro_func(ptr %arg)
  store i64 %val, ptr %loc
  %cmp = icmp ne i64 %call, 0
  br i1 %cmp, label %then, label %else

then:
  ret i64 %call

else:
  %loaded = load i64, ptr %loc
  ret i64 %loaded
}

; ============================================================================
; Same three call patterns as the previous variants, but the store of %val
; into the pre-existing alloca happens BEFORE the call instead of after it.
; The expected outcome (number of NEW allocas generated) is the same as in
; the previous variants: the position of the store relative to the call
; does not change the picker's decision.
; ============================================================================

; Variant of @call_rw_one_use_with_alloca with the store before the call.
; Same outcome: no NEW alloca generated, only the pre-existing one survives.
;
; CHECK-LABEL: define i64 @call_rw_one_use_with_alloca_store_before(
; CHECK: alloca i64
; CHECK-NOT: alloca

define i64 @call_rw_one_use_with_alloca_store_before(ptr %arg, i64 %val) {
entry:
  %loc = alloca i64
  store i64 %val, ptr %loc
  %call = call i64 @rw_func(ptr %arg)
  %cmp = icmp ne i64 %call, 0
  br i1 %cmp, label %then, label %else

then:
  %loaded_then = load i64, ptr %loc
  ret i64 %loaded_then

else:
  %loaded_else = load i64, ptr %loc
  ret i64 %loaded_else
}

; Variant of @call_rw_two_uses_with_alloca with the store before the call.
; Same outcome: 2 allocas total (pre-existing + new for the picked call).
;
; CHECK-LABEL: define i64 @call_rw_two_uses_with_alloca_store_before(
; CHECK: alloca i64
; CHECK: alloca i64
; CHECK-NOT: alloca

define i64 @call_rw_two_uses_with_alloca_store_before(ptr %arg, i64 %val) {
entry:
  %loc = alloca i64
  store i64 %val, ptr %loc
  %call = call i64 @rw_func(ptr %arg)
  %cmp = icmp ne i64 %call, 0
  br i1 %cmp, label %then, label %else

then:
  ret i64 %call

else:
  %loaded = load i64, ptr %loc
  ret i64 %loaded
}

; Variant of @call_ro_two_uses_with_alloca with the store before the call.
; Same outcome: no NEW alloca generated, only the pre-existing one survives.
;
; CHECK-LABEL: define i64 @call_ro_two_uses_with_alloca_store_before(
; CHECK: alloca i64
; CHECK-NOT: alloca

define i64 @call_ro_two_uses_with_alloca_store_before(ptr %arg, i64 %val) {
entry:
  %loc = alloca i64
  store i64 %val, ptr %loc
  %call = call i64 @ro_func(ptr %arg)
  %cmp = icmp ne i64 %call, 0
  br i1 %cmp, label %then, label %else

then:
  ret i64 %call

else:
  %loaded = load i64, ptr %loc
  ret i64 %loaded
}

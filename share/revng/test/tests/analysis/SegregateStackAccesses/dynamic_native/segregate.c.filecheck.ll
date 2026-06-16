;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; This file mixes checks for what we expect to happen on a x86-64 binarya and an
; i386 binary. In this case, the i386 binary is interesting due to SystemV ABI
; making heavy use of stack arguments.

; raw_primitives_on_registers

CHECK-x86_64:     define i64 @local_raw_primitives_on_registers(i64 %[[ARG1:.*]], i64 %[[ARG2:.*]]) [[IGN:.*]] {
CHECK-x86_64-DAG:   add i64 [[IGN:.*]]%[[ARG1]]
CHECK-x86_64-DAG:   add i64 [[IGN:.*]]%[[ARG2]]
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_raw_primitives_on_registers() [[IGN:.*]] {
CHECK-x86_64-DAG:   = call i64 @local_raw_primitives_on_registers(i64 2, i64 1)
CHECK-x86_64:     }

; raw_pointers_on_registers

CHECK-x86_64:     define i64 @local_raw_pointers_on_registers(i64 %[[ARG1:.*]], i64 %[[ARG2:.*]]) [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[ARG1_PTR:.*]] = inttoptr i64 %[[ARG1]] to ptr
CHECK-x86_64-DAG:   load i64, ptr %[[ARG1_PTR]]
CHECK-x86_64-DAG:   %[[ARG2_PTR:.*]] = inttoptr i64 %[[ARG2]] to ptr
CHECK-x86_64-DAG:   load i64, ptr %[[ARG2_PTR]]
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_raw_pointers_on_registers() [[IGN:.*]] {
CHECK-x86_64-DAG:   = call i64 @local_raw_pointers_on_registers(i64 [[ARG:.*]], i64 [[ARG]])
CHECK-x86_64:     }

; raw_primitives_on_stack

CHECK-x86_64:     define i64 @local_raw_primitives_on_stack(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG:.*]]) [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[STACK_ARG8:.*]] = add i64 %[[STACK_ARG]], 8
CHECK-x86_64-DAG:   %[[STACK_ARG8_PTR:.*]] = inttoptr i64 %[[STACK_ARG8]] to ptr
CHECK-x86_64-DAG:   load i64, ptr %[[STACK_ARG8_PTR]]
CHECK-x86_64-DAG:   %[[STACK_ARG_PTR:.*]] = inttoptr i64 %[[STACK_ARG]] to ptr
CHECK-x86_64-DAG:   load i64, ptr %[[STACK_ARG_PTR]]
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_raw_primitives_on_stack() [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[STACK:.*]] = alloca [16 x i8]
CHECK-x86_64-DAG:   %[[STACK_INT:.*]] = ptrtoint ptr %[[STACK]] to i64
CHECK-x86_64-DAG:   %[[STACK_INT_8:.*]] = add i64 %[[STACK_INT]], 8
CHECK-x86_64-DAG:   %[[STACK_8:.*]] = inttoptr i64 %[[STACK_INT_8]] to ptr
CHECK-x86_64-DAG:   store i64 8, ptr %[[STACK_8]]
CHECK-x86_64-DAG:   store i64 7, ptr %[[STACK]]
CHECK-x86_64-DAG:   = call i64 @local_raw_primitives_on_stack(i64 4, i64 3, i64 2, i64 1, i64 5, i64 6, i64 %[[STACK_INT]])
CHECK-x86_64:     }

; cabi_primitives_on_registers

CHECK-x86_64:     define i64 @local_cabi_primitives_on_registers(i64 %[[ARG1:.*]], i64 %[[ARG2:.*]]) [[IGN:.*]] {
CHECK-x86_64-DAG:   add i64 [[IGN:.*]]%[[ARG1]]
CHECK-x86_64-DAG:   add i64 [[IGN:.*]]%[[ARG2]]
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_cabi_primitives_on_registers() [[IGN:.*]] {
CHECK-x86_64-DAG:   = call i64 @local_cabi_primitives_on_registers(i64 1, i64 2)
CHECK-x86_64:     }

CHECK-i386:     define i32 @local_cabi_primitives_on_registers(i32 %[[A:.*]], i32 %[[B:.*]]) [[IGN:.*]] {
CHECK-i386-DAG:   %[[T1:.*]] = add i32 {{.*}}, %[[A]]
CHECK-i386-DAG:   %[[T2:.*]] = add i32 %[[T1]], %[[B]]
CHECK-i386:       ret i32 %[[T2]]
CHECK-i386:     }

CHECK-i386: define i32 @local_call_cabi_primitives_on_registers() [[IGN:.*]] {
CHECK-i386:   %[[CALL:.*]] = call i32 @local_cabi_primitives_on_registers(i32 1, i32 2)
CHECK-i386:   %[[RET:.*]] = add i32 %[[CALL]], {{.*}}
CHECK-i386:   ret i32 %[[RET]]
CHECK-i386: }

; cabi_primitives_on_stack

CHECK-x86_64:     define i64 @local_cabi_primitives_on_stack(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG1:.*]], i64 %[[STACK_ARG2:.*]]) [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[IGN:.*]] = add i64 %[[IGN:.*]]%[[STACK_ARG1]]
CHECK-x86_64-DAG:   %[[IGN:.*]] = add i64 %[[IGN:.*]]%[[STACK_ARG2]]
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_cabi_primitives_on_stack() [[IGN:.*]] {
CHECK-x86_64-DAG:   = call i64 @local_cabi_primitives_on_stack(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 [[SCALAR1:.*]], i64 [[SCALAR2:.*]])
CHECK-x86_64:     }

CHECK-i386: define i32 @local_cabi_primitives_on_stack(i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[G:.*]], i32 %[[H:.*]]) [[IGN:.*]] {
CHECK-i386:   %[[T1:.*]] = add i32 {{.*}}, %[[G]]
CHECK-i386:   %[[T2:.*]] = add i32 %[[T1]], %[[H]]
CHECK-i386:   ret i32 %[[T2]]
CHECK-i386: }

CHECK-i386: define i32 @local_call_cabi_primitives_on_stack() [[IGN:.*]] {
CHECK-i386:   %[[CALL:.*]] = call i32 @local_cabi_primitives_on_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8)
CHECK-i386:   %[[RET:.*]] = add i32 %[[CALL]], {{.*}}
CHECK-i386:   ret i32 %[[RET]]
CHECK-i386: }

; cabi_aggregate_on_registers

CHECK-x86_64:     define i64 @local_cabi_aggregate_on_registers(i64 %[[ARG1:.*]]) [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[FIELD1_PTR:.*]] = inttoptr i64 %[[ARG1]] to ptr
CHECK-x86_64-DAG:   load i64, ptr %[[FIELD1_PTR]]
CHECK-x86_64-DAG:   %[[FIELD2_ADDR:.*]] = add i64 %[[ARG1]], 8
CHECK-x86_64-DAG:   %[[FIELD2_PTR:.*]] = inttoptr i64 %[[FIELD2_ADDR]] to ptr
CHECK-x86_64-DAG:   load i64, ptr %[[FIELD2_PTR]], align 8
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_cabi_aggregate_on_registers() [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[STACK:.*]] = alloca [16 x i8]
CHECK-x86_64-DAG:   %[[STACK_INT:.*]] = ptrtoint ptr %[[STACK]] to i64
CHECK-x86_64-DAG:   store i64 1, ptr %[[STACK]]
CHECK-x86_64-DAG:   %[[STACK_INT_8:.*]] = add i64 %[[STACK_INT]], 8
CHECK-x86_64-DAG:   %[[STACK_8:.*]] = inttoptr i64 %[[STACK_INT_8]] to ptr
CHECK-x86_64-DAG:   store i64 2, ptr %[[STACK_8]]
CHECK-x86_64-DAG:   = call i64 @local_cabi_aggregate_on_registers(i64 %[[STACK_INT]])
CHECK-x86_64:     }

CHECK-i386: define i32 @local_cabi_aggregate_on_registers(i32 %[[PTR:.*]]) [[IGN:.*]] {
CHECK-i386:   %[[A_PTR:.*]] = inttoptr i32 %[[PTR]] to ptr
CHECK-i386:   %[[A:.*]] = load i32, ptr %[[A_PTR]]
CHECK-i386:   %[[T1:.*]] = add i32 {{.*}}, %[[A]]
CHECK-i386:   %[[B_OFF:.*]] = add i32 %[[PTR]], 4
CHECK-i386:   %[[B_PTR:.*]] = inttoptr i32 %[[B_OFF]] to ptr
CHECK-i386:   %[[B:.*]] = load i32, ptr %[[B_PTR]]
CHECK-i386:   %[[T2:.*]] = add i32 %[[T1]], %[[B]]
CHECK-i386:   ret i32 %[[T2]]
CHECK-i386: }

CHECK-i386: define i32 @local_call_cabi_aggregate_on_registers() [[IGN:.*]] {
CHECK-i386:   %[[SLOT:.*]] = alloca [8 x i8]
CHECK-i386:   %[[SLOT_INT:.*]] = ptrtoint ptr %[[SLOT]] to i32
CHECK-i386:   %[[B_OFF:.*]] = add i32 %[[SLOT_INT]], 4
CHECK-i386:   %[[B_PTR:.*]] = inttoptr i32 %[[B_OFF]] to ptr
CHECK-i386:   store i32 2, ptr %[[B_PTR]]
CHECK-i386:   %[[A_PTR:.*]] = inttoptr i32 %[[SLOT_INT]] to ptr
CHECK-i386:   store i32 1, ptr %[[A_PTR]]
CHECK-i386:   %[[CALL:.*]] = call i32 @local_cabi_aggregate_on_registers(i32 %[[SLOT_INT]])
CHECK-i386:   %[[RET:.*]] = add i32 %[[CALL]], {{.*}}
CHECK-i386:   ret i32 %[[RET]]
CHECK-i386: }

; cabi_aggregate_on_stack

CHECK-x86_64:     define i64 @local_cabi_aggregate_on_stack(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG:.*]]) [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[FIELD1_PTR:.*]] = inttoptr i64 %[[STACK_ARG]] to ptr
CHECK-x86_64-DAG:   load i64, ptr %[[FIELD1_PTR]]
CHECK-x86_64-DAG:   %[[FIELD2_ADDR:.*]] = add i64 %[[STACK_ARG]], 8
CHECK-x86_64-DAG:   %[[FIELD2_PTR:.*]] = inttoptr i64 %[[FIELD2_ADDR]] to ptr
CHECK-x86_64-DAG:   load i64, ptr %[[FIELD2_PTR]]
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_cabi_aggregate_on_stack() [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[STACK:.*]] = alloca [16 x i8]
CHECK-x86_64-DAG:   %[[STACK_INT:.*]] = ptrtoint ptr %[[STACK]] to i64
CHECK-x86_64-DAG:   store i64 1, ptr %[[STACK]]
CHECK-x86_64-DAG:   %[[STACK_INT_8:.*]] = add i64 %[[STACK_INT]], 8
CHECK-x86_64-DAG:   %[[STACK_8:.*]] = inttoptr i64 %[[STACK_INT_8]] to ptr
CHECK-x86_64-DAG:   store i64 2, ptr %[[STACK_8]]
CHECK-x86_64-DAG:   = call i64 @local_cabi_aggregate_on_stack(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 %[[STACK_INT]])
CHECK-x86_64:     }

CHECK-i386: define i32 @local_cabi_aggregate_on_stack(i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[PTR:.*]]) [[IGN:.*]] {
CHECK-i386:   %[[A_PTR:.*]] = inttoptr i32 %[[PTR]] to ptr
CHECK-i386:   %[[A:.*]] = load i32, ptr %[[A_PTR]]
CHECK-i386:   %[[T1:.*]] = add i32 {{.*}}, %[[A]]
CHECK-i386:   %[[B_OFF:.*]] = add i32 %[[PTR]], 4
CHECK-i386:   %[[B_PTR:.*]] = inttoptr i32 %[[B_OFF]] to ptr
CHECK-i386:   %[[B:.*]] = load i32, ptr %[[B_PTR]]
CHECK-i386:   %[[T2:.*]] = add i32 %[[T1]], %[[B]]
CHECK-i386:   ret i32 %[[T2]]
CHECK-i386: }

CHECK-i386: define i32 @local_call_cabi_aggregate_on_stack() [[IGN:.*]] {
CHECK-i386:   %[[SLOT:.*]] = alloca [8 x i8]
CHECK-i386:   %[[SLOT_INT:.*]] = ptrtoint ptr %[[SLOT]] to i32
CHECK-i386:   %[[B_OFF:.*]] = add i32 %[[SLOT_INT]], 4
CHECK-i386:   %[[B_PTR:.*]] = inttoptr i32 %[[B_OFF]] to ptr
CHECK-i386:   store i32 2, ptr %[[B_PTR]]
CHECK-i386:   %[[A_PTR:.*]] = inttoptr i32 %[[SLOT_INT]] to ptr
CHECK-i386:   store i32 1, ptr %[[A_PTR]]
CHECK-i386:   %[[CALL:.*]] = call i32 @local_cabi_aggregate_on_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 %[[SLOT_INT]])
CHECK-i386:   %[[RET:.*]] = add i32 %[[CALL]], {{.*}}
CHECK-i386:   ret i32 %[[RET]]
CHECK-i386: }

; cabi_aggregate_on_stack_and_registers

CHECK-x86_64:     define i64 @local_cabi_aggregate_on_stack_and_registers(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG:.*]]) [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[FIELD1_PTR:.*]] = inttoptr i64 %[[STACK_ARG]] to ptr
CHECK-x86_64-DAG:   load i64, ptr %[[FIELD1_PTR]]
CHECK-x86_64-DAG:   %[[FIELD2_ADDR:.*]] = add i64 %[[STACK_ARG]], 8
CHECK-x86_64-DAG:   %[[FIELD2_PTR:.*]] = inttoptr i64 %[[FIELD2_ADDR]] to ptr
CHECK-x86_64-DAG:   load i64, ptr %[[FIELD2_PTR]]
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_cabi_aggregate_on_stack_and_registers() [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[STACK:.*]] = alloca [16 x i8]
CHECK-x86_64-DAG:   %[[STACK_INT:.*]] = ptrtoint ptr %[[STACK]] to i64
CHECK-x86_64-DAG:   store i64 1, ptr %[[STACK]]
CHECK-x86_64-DAG:   %[[STACK_INT_8:.*]] = add i64 %[[STACK_INT]], 8
CHECK-x86_64-DAG:   %[[STACK_8:.*]] = inttoptr i64 %[[STACK_INT_8]] to ptr
CHECK-x86_64-DAG:   store i64 2, ptr %[[STACK_8]]
CHECK-x86_64-DAG:   = call i64 @local_cabi_aggregate_on_stack_and_registers(i64 1, i64 2, i64 3, i64 4, i64 5, i64 %[[STACK_INT]])
CHECK-x86_64:     }

CHECK-i386: define i32 @local_cabi_aggregate_on_stack_and_registers(i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[IGN:.*]], i32 %[[PTR:.*]]) [[IGN:.*]] {
CHECK-i386:   %[[A_PTR:.*]] = inttoptr i32 %[[PTR]] to ptr
CHECK-i386:   %[[A:.*]] = load i32, ptr %[[A_PTR]]
CHECK-i386:   %[[T1:.*]] = add i32 {{.*}}, %[[A]]
CHECK-i386:   %[[B_OFF:.*]] = add i32 %[[PTR]], 4
CHECK-i386:   %[[B_PTR:.*]] = inttoptr i32 %[[B_OFF]] to ptr
CHECK-i386:   %[[B:.*]] = load i32, ptr %[[B_PTR]]
CHECK-i386:   %[[T2:.*]] = add i32 %[[T1]], %[[B]]
CHECK-i386:   ret i32 %[[T2]]
CHECK-i386: }

CHECK-i386: define i32 @local_call_cabi_aggregate_on_stack_and_registers() [[IGN:.*]] {
CHECK-i386:   %[[SLOT:.*]] = alloca [8 x i8]
CHECK-i386:   %[[SLOT_INT:.*]] = ptrtoint ptr %[[SLOT]] to i32
CHECK-i386:   %[[B_OFF:.*]] = add i32 %[[SLOT_INT]], 4
CHECK-i386:   %[[B_PTR:.*]] = inttoptr i32 %[[B_OFF]] to ptr
CHECK-i386:   store i32 2, ptr %[[B_PTR]]
CHECK-i386:   %[[A_PTR:.*]] = inttoptr i32 %[[SLOT_INT]] to ptr
CHECK-i386:   store i32 1, ptr %[[A_PTR]]
CHECK-i386:   %[[CALL:.*]] = call i32 @local_cabi_aggregate_on_stack_and_registers(i32 1, i32 2, i32 3, i32 4, i32 5, i32 %[[SLOT_INT]])
CHECK-i386:   %[[RET:.*]] = add i32 %[[CALL]], {{.*}}
CHECK-i386:   ret i32 %[[RET]]
CHECK-i386: }

; raw_return_small_aggregate

CHECK-x86_64:     define <{ i64, i64 }> @local_raw_return_small_aggregate() [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[RESULT:.*]] = call <{ i64, i64 }> @struct_initializer(i64 124, i64 123)
CHECK-x86_64-DAG:   ret <{ i64, i64 }> %[[RESULT]]
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_raw_return_small_aggregate() [[IGN:.*]] {
CHECK-x86_64:       %[[RESULT:.*]] = call <{ i64, i64 }> @local_raw_return_small_aggregate()
CHECK-x86_64-DAG:   call i64 @OpaqueExtractvalue(<{ i64, i64 }> %[[RESULT]], i64 1)
CHECK-x86_64:     }

; cabi_return_small_aggregate

CHECK-x86_64:     define [16 x i8] @local_cabi_return_small_aggregate() [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA:.*]] = alloca [16 x i8]
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_INT:.*]] = ptrtoint ptr %[[RETURN_ALLOCA]] to i64
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_INT_8:.*]] = add i64 %[[RETURN_ALLOCA_INT]], 8
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_8:.*]] = inttoptr i64 %[[RETURN_ALLOCA_INT_8]] to ptr
CHECK-x86_64-DAG:   store i64 124, ptr %[[RETURN_ALLOCA]]
CHECK-x86_64-DAG:   store i64 123, ptr %[[RETURN_ALLOCA_8]]
CHECK-x86_64-DAG:   %[[TO_RETURN:.*]] = load [16 x i8], ptr %[[RETURN_ALLOCA]]
CHECK-x86_64-DAG:   ret [16 x i8] %[[TO_RETURN]]
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_cabi_return_small_aggregate() [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA:.*]] = alloca [16 x i8]
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_INT:.*]] = ptrtoint ptr %[[RETURN_ALLOCA]] to i64
CHECK-x86_64-DAG:   %[[RETURN_VALUE:.*]] = call [16 x i8] @local_cabi_return_small_aggregate()
CHECK-x86_64-DAG:   store [16 x i8] %[[RETURN_VALUE]], ptr %[[RETURN_ALLOCA]]
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_INT_8:.*]] = add i64 %[[RETURN_ALLOCA_INT]], 8
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_8:.*]] = inttoptr i64 %[[RETURN_ALLOCA_INT_8]] to ptr
CHECK-x86_64-DAG:   %[[TO_RETURN:.*]] = load i64, ptr %[[RETURN_ALLOCA_8]]
CHECK-x86_64:     }

CHECK-i386:     define [8 x i8] @local_cabi_return_small_aggregate() [[IGN:.*]] {
CHECK-i386:       %[[SLOT:.*]] = alloca [8 x i8]
CHECK-i386:       %[[SLOT_INT:.*]] = ptrtoint ptr %[[SLOT]] to i32
CHECK-i386:       %[[SLOT_PTR:.*]] = inttoptr i32 %[[SLOT_INT]] to ptr
CHECK-i386:       %[[BASE:.*]] = load i32, ptr %[[SLOT_PTR]]
CHECK-i386-DAG:   store i32 124, ptr {{.*}}
CHECK-i386-DAG:   %[[OFF:.*]] = add i32 %[[BASE]], 4
CHECK-i386-DAG:   store i32 123, ptr {{.*}}
CHECK-i386:       %[[RV:.*]] = load [8 x i8], ptr %[[SLOT]]
CHECK-i386:       ret [8 x i8] %[[RV]]
CHECK-i386:     }

CHECK-i386: define i32 @local_call_cabi_return_small_aggregate() [[IGN:.*]] {
CHECK-i386:   %[[SLOT:.*]] = alloca [8 x i8]
CHECK-i386:   %[[RV:.*]] = call [8 x i8] @local_cabi_return_small_aggregate()
CHECK-i386:   store [8 x i8] %[[RV]], ptr %[[SLOT]]
CHECK-i386:   %[[SLOT_INT:.*]] = ptrtoint ptr %[[SLOT]] to i32
CHECK-i386:   %[[SLOT_PTR:.*]] = inttoptr i32 %[[SLOT_INT]] to ptr
CHECK-i386:   %[[B:.*]] = load i32, ptr %[[SLOT_PTR]]
CHECK-i386:   %[[RET:.*]] = add i32 {{.*}}%[[B]]
CHECK-i386:   ret i32 %[[RET]]
CHECK-i386: }


; cabi_return_big_aggregate

CHECK-x86_64:     define [64 x i8] @local_cabi_return_big_aggregate() [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA:.*]] = alloca [64 x i8]
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_INT:.*]] = ptrtoint ptr %[[RETURN_ALLOCA]] to i64
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_INT_16:.*]] = add i64 %[[RETURN_ALLOCA_INT]], 16
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_16:.*]] = inttoptr i64 %[[RETURN_ALLOCA_INT_16]] to ptr
CHECK-x86_64-DAG:   store i64 123, ptr %[[RETURN_ALLOCA_16]]
CHECK-x86_64-DAG:   %[[TO_RETURN:.*]] = load [64 x i8], ptr %[[RETURN_ALLOCA]]
CHECK-x86_64-DAG:   ret [64 x i8] %[[TO_RETURN]]
CHECK-x86_64:     }

CHECK-x86_64:     define i64 @local_call_cabi_return_big_aggregate() [[IGN:.*]] {
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA:.*]] = alloca [64 x i8]
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_INT:.*]] = ptrtoint ptr %[[RETURN_ALLOCA]] to i64
CHECK-x86_64-DAG:   %[[RETURN_VALUE:.*]] = call [64 x i8] @local_cabi_return_big_aggregate()
CHECK-x86_64-DAG:   store [64 x i8] %[[RETURN_VALUE]], ptr %[[RETURN_ALLOCA]]
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_INT_16:.*]] = add i64 %[[RETURN_ALLOCA_INT]], 16
CHECK-x86_64-DAG:   %[[RETURN_ALLOCA_16:.*]] = inttoptr i64 %[[RETURN_ALLOCA_INT_16]] to ptr
CHECK-x86_64-DAG:   %[[TO_RETURN:.*]] = load i64, ptr %[[RETURN_ALLOCA_16]]
CHECK-x86_64:     }

CHECK-i386: define [28 x i8] @local_cabi_return_big_aggregate() [[IGN:.*]] {
CHECK-i386:   %[[SLOT:.*]] = alloca [28 x i8]
CHECK-i386:   %[[SLOT_INT:.*]] = ptrtoint ptr %[[SLOT]] to i32
CHECK-i386:   %[[SLOT_PTR:.*]] = inttoptr i32 %[[SLOT_INT]] to ptr
CHECK-i386:   %[[BASE:.*]] = load i32, ptr %[[SLOT_PTR]]
CHECK-i386:   %[[OFF:.*]] = add i32 %[[BASE]], 8
CHECK-i386:   store i32 123, ptr {{.*}}
CHECK-i386:   %[[RV:.*]] = load [28 x i8], ptr %[[SLOT]]
CHECK-i386:   ret [28 x i8] %[[RV]]
CHECK-i386: }

CHECK-i386: define i32 @local_call_cabi_return_big_aggregate() [[IGN:.*]] {
CHECK-i386:   %[[SLOT:.*]] = alloca [28 x i8]
CHECK-i386:   %[[RV:.*]] = call [28 x i8] @local_cabi_return_big_aggregate()
CHECK-i386:   store [28 x i8] %[[RV]], ptr %[[SLOT]]
CHECK-i386:   %[[SLOT_INT:.*]] = ptrtoint ptr %[[SLOT]] to i32
CHECK-i386:   %[[OFF:.*]] = add i32 %[[SLOT_INT]], 4
CHECK-i386:   %[[SLOT_PTR:.*]] = inttoptr i32 %[[OFF]] to ptr
CHECK-i386:   %[[D:.*]] = load i32, ptr %[[SLOT_PTR]]
CHECK-i386:   %[[RET:.*]] = add i32 {{.*}}%[[D]]
CHECK-i386:   ret i32 %[[RET]]
CHECK-i386: }

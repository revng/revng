;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

CHECK: define i64 @local_raw_primitives_on_registers(i64 %[[ARG1:.*]], i64 %[[ARG2:.*]]) [[IGN:.*]] {
CHECK-DAG: add i64 [[IGN:.*]]%[[ARG1]]
CHECK-DAG: add i64 [[IGN:.*]]%[[ARG2]]
CHECK: }

CHECK: define i64 @local_call_raw_primitives_on_registers() [[IGN:.*]] {
CHECK-DAG:   = call i64 @local_raw_primitives_on_registers(i64 2, i64 1)
CHECK: }

CHECK: define i64 @local_raw_pointers_on_registers(i64 %[[ARG1:.*]], i64 %[[ARG2:.*]]) [[IGN:.*]] {
CHECK-DAG: %[[ARG1_PTR:.*]] = inttoptr i64 %[[ARG1]] to ptr
CHECK-DAG: load i64, ptr %[[ARG1_PTR:.*]]
CHECK-DAG: %[[ARG2_PTR:.*]] = inttoptr i64 %[[ARG2]] to ptr
CHECK-DAG: load i64, ptr %[[ARG2_PTR]]
CHECK: }

CHECK: define i64 @local_call_raw_pointers_on_registers() [[IGN:.*]] {
CHECK-DAG:   = call i64 @local_raw_pointers_on_registers(i64 [[ARG:.*]], i64 [[ARG]])
CHECK: }

CHECK: define i64 @local_raw_primitives_on_stack(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG:.*]]) [[IGN:.*]] {
CHECK-DAG:   %[[STACK_ARG8:.*]] = add i64 %[[STACK_ARG]], 8
CHECK-DAG:   %[[STACK_ARG8_PTR:.*]] = inttoptr i64 %[[STACK_ARG8]] to ptr
CHECK-DAG:   load i64, ptr %[[STACK_ARG8_PTR:.*]]
CHECK-DAG:   %[[STACK_ARG_PTR:.*]] = inttoptr i64 %[[STACK_ARG]] to ptr
CHECK-DAG:   load i64, ptr %[[STACK_ARG_PTR]]
CHECK: }

CHECK: define i64 @local_call_raw_primitives_on_stack() [[IGN:.*]] {
CHECK-DAG:   %[[STACK:.*]] = alloca [16 x i8]
CHECK-DAG:   %[[STACK_INT:.*]] = ptrtoint ptr %[[STACK]] to i64
CHECK-DAG:   %[[STACK_INT_8:.*]] = add i64 %[[STACK_INT]], 8
CHECK-DAG:   %[[STACK_8:.*]] = inttoptr i64 %[[STACK_INT_8]] to ptr
CHECK-DAG:   store i64 8, ptr %[[STACK_8]]
CHECK-DAG:   store i64 7, ptr %[[STACK]]
CHECK-DAG:   = call i64 @local_raw_primitives_on_stack(i64 4, i64 3, i64 2, i64 1, i64 5, i64 6, i64 %[[STACK_INT]])
CHECK: }

CHECK: define i64 @local_cabi_primitives_on_registers(i64 %[[ARG1:.*]], i64 %[[ARG2:.*]]) [[IGN:.*]] {
CHECK-DAG: add i64 [[IGN:.*]]%[[ARG1]]
CHECK-DAG: add i64 [[IGN:.*]]%[[ARG2]]
CHECK: }

CHECK: define i64 @local_call_cabi_primitives_on_registers() [[IGN:.*]] {
CHECK-DAG:   = call i64 @local_cabi_primitives_on_registers(i64 1, i64 2)
CHECK: }

CHECK: define i64 @local_cabi_primitives_on_stack(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG1:.*]], i64 %[[STACK_ARG2:.*]]) [[IGN:.*]] {
CHECK-DAG:   %[[IGN:.*]] = add i64 %[[IGN:.*]]%[[STACK_ARG1]]
CHECK-DAG:   %[[IGN:.*]] = add i64 %[[IGN:.*]]%[[STACK_ARG2]]
CHECK: }

// WIP: we get undef here, wrong!
CHECK: define i64 @local_call_cabi_primitives_on_stack() [[IGN:.*]] {
CHECK-DAG:   = call i64 @local_cabi_primitives_on_stack(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 [[SCALAR1:.*]], i64 [[SCALAR2:.*]])
CHECK: }

CHECK: define i64 @local_cabi_aggregate_on_registers(i64 %[[ARG1:.*]]) [[IGN:.*]] {
CHECK-DAG:   %[[FIELD1_PTR:.*]] = inttoptr i64 %[[ARG1]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD1_PTR]]
CHECK-DAG:   %[[FIELD2_ADDR:.*]] = add i64 %[[ARG1]], 8
CHECK-DAG:   %[[FIELD2_PTR:.*]] = inttoptr i64 %[[FIELD2_ADDR]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD2_PTR]], align 8
CHECK: }

CHECK: define i64 @local_call_cabi_aggregate_on_registers() [[IGN:.*]] {
CHECK-DAG:   %[[STACK:.*]] = alloca [16 x i8]
CHECK-DAG:   %[[STACK_INT:.*]] = ptrtoint ptr %[[STACK]] to i64
CHECK-DAG:   store i64 1, ptr %[[STACK]]
CHECK-DAG:   %[[STACK_INT_8:.*]] = add i64 %[[STACK_INT]], 8
CHECK-DAG:   %[[STACK_8:.*]] = inttoptr i64 %[[STACK_INT_8]] to ptr
CHECK-DAG:   store i64 2, ptr %[[STACK_8]]
CHECK-DAG:   = call i64 @local_cabi_aggregate_on_registers(i64 %[[STACK_INT]])
CHECK: }

CHECK: define i64 @local_cabi_aggregate_on_stack(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG:.*]]) [[IGN:.*]] {
CHECK-DAG:   %[[FIELD1_PTR:.*]] = inttoptr i64 %[[STACK_ARG]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD1_PTR]]
CHECK-DAG:   %[[FIELD2_ADDR:.*]] = add i64 %[[STACK_ARG]], 8
CHECK-DAG:   %[[FIELD2_PTR:.*]] = inttoptr i64 %[[FIELD2_ADDR]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD2_PTR]]
CHECK: }

CHECK: define i64 @local_call_cabi_aggregate_on_stack() [[IGN:.*]] {
CHECK-DAG:   %[[STACK:.*]] = alloca [16 x i8]
CHECK-DAG:   %[[STACK_INT:.*]] = ptrtoint ptr %[[STACK]] to i64
CHECK-DAG:   store i64 1, ptr %[[STACK]]
CHECK-DAG:   %[[STACK_INT_8:.*]] = add i64 %[[STACK_INT]], 8
CHECK-DAG:   %[[STACK_8:.*]] = inttoptr i64 %[[STACK_INT_8]] to ptr
CHECK-DAG:   store i64 2, ptr %[[STACK_8]]
CHECK-DAG:   = call i64 @local_cabi_aggregate_on_stack(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 %[[STACK_INT]])
CHECK: }

CHECK: define i64 @local_cabi_aggregate_on_stack_and_registers(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG:.*]]) [[IGN:.*]] {
CHECK-DAG:   %[[FIELD1_PTR:.*]] = inttoptr i64 %[[STACK_ARG]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD1_PTR]]
CHECK-DAG:   %[[FIELD2_ADDR:.*]] = add i64 %[[STACK_ARG]], 8
CHECK-DAG:   %[[FIELD2_PTR:.*]] = inttoptr i64 %[[FIELD2_ADDR]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD2_PTR]]
CHECK: }

CHECK: define i64 @local_call_cabi_aggregate_on_stack_and_registers() [[IGN:.*]] {
CHECK-DAG:   %[[STACK:.*]] = alloca [16 x i8]
CHECK-DAG:   %[[STACK_INT:.*]] = ptrtoint ptr %[[STACK]] to i64
CHECK-DAG:   store i64 1, ptr %[[STACK]]
CHECK-DAG:   %[[STACK_INT_8:.*]] = add i64 %[[STACK_INT]], 8
CHECK-DAG:   %[[STACK_8:.*]] = inttoptr i64 %[[STACK_INT_8]] to ptr
CHECK-DAG:   store i64 2, ptr %[[STACK_8]]
CHECK-DAG:   = call i64 @local_cabi_aggregate_on_stack_and_registers(i64 1, i64 2, i64 3, i64 4, i64 5, i64 %[[STACK_INT]])
CHECK: }

CHECK: define <{ i64, i64 }> @local_raw_return_small_aggregate() [[IGN:.*]] {
CHECK-DAG:   %[[RESULT:.*]] = call <{ i64, i64 }> @struct_initializer(i64 124, i64 123)
CHECK-DAG:   ret <{ i64, i64 }> %[[RESULT]]
CHECK: }

CHECK: define i64 @local_call_raw_return_small_aggregate() [[IGN:.*]] {
CHECK:   %[[RESULT:.*]] = call <{ i64, i64 }> @local_raw_return_small_aggregate()
CHECK-DAG:   call i64 @OpaqueExtractvalue(<{ i64, i64 }> %[[RESULT]], i64 1)
CHECK: }

CHECK: define [16 x i8] @local_cabi_return_small_aggregate() [[IGN:.*]] {
CHECK-DAG:   %[[RETURN_ALLOCA:.*]] = alloca [16 x i8]
CHECK-DAG:   %[[RETURN_ALLOCA_INT:.*]] = ptrtoint ptr %[[RETURN_ALLOCA]] to i64
CHECK-DAG:   %[[RETURN_ALLOCA_INT_8:.*]] = add i64 %[[RETURN_ALLOCA_INT]], 8
CHECK-DAG:   %[[RETURN_ALLOCA_8:.*]] = inttoptr i64 %[[RETURN_ALLOCA_INT_8]] to ptr
CHECK-DAG:   store i64 124, ptr %[[RETURN_ALLOCA]]
CHECK-DAG:   store i64 123, ptr %[[RETURN_ALLOCA_8]]
CHECK-DAG:   %[[TO_RETURN:.*]] = load [16 x i8], ptr %[[RETURN_ALLOCA]]
CHECK-DAG:   ret [16 x i8] %[[TO_RETURN]]
CHECK: }

CHECK: define i64 @local_call_cabi_return_small_aggregate() [[IGN:.*]] {
CHECK-DAG:   %[[RETURN_ALLOCA:.*]] = alloca [16 x i8]
CHECK-DAG:   %[[RETURN_ALLOCA_INT:.*]] = ptrtoint ptr %[[RETURN_ALLOCA]] to i64
CHECK-DAG:   %[[RETURN_VALUE:.*]] = call [16 x i8] @local_cabi_return_small_aggregate()
CHECK-DAG:   store [16 x i8] %[[RETURN_VALUE]], ptr %[[RETURN_ALLOCA]]
CHECK-DAG:   %[[RETURN_ALLOCA_INT_8:.*]] = add i64 %[[RETURN_ALLOCA_INT]], 8
CHECK-DAG:   %[[RETURN_ALLOCA_8:.*]] = inttoptr i64 %[[RETURN_ALLOCA_INT_8]] to ptr
CHECK-DAG:   %[[TO_RETURN:.*]] = load i64, ptr %[[RETURN_ALLOCA_8]]
CHECK: }

CHECK: define [64 x i8] @local_cabi_return_big_aggregate() [[IGN:.*]] {
CHECK-DAG:   %[[RETURN_ALLOCA:.*]] = alloca [64 x i8]
CHECK-DAG:   %[[RETURN_ALLOCA_INT:.*]] = ptrtoint ptr %[[RETURN_ALLOCA]] to i64
CHECK-DAG:   %[[RETURN_ALLOCA_INT_16:.*]] = add i64 %[[RETURN_ALLOCA_INT]], 16
CHECK-DAG:   %[[RETURN_ALLOCA_16:.*]] = inttoptr i64 %[[RETURN_ALLOCA_INT_16]] to ptr
CHECK-DAG:   store i64 123, ptr %[[RETURN_ALLOCA_16]]
CHECK-DAG:   %[[TO_RETURN:.*]] = load [64 x i8], ptr %[[RETURN_ALLOCA]]
CHECK-DAG:   ret [64 x i8] %[[TO_RETURN]]
CHECK: }

CHECK: define i64 @local_call_cabi_return_big_aggregate() [[IGN:.*]] {
CHECK-DAG:   %[[RETURN_ALLOCA:.*]] = alloca [64 x i8]
CHECK-DAG:   %[[RETURN_ALLOCA_INT:.*]] = ptrtoint ptr %[[RETURN_ALLOCA]] to i64
CHECK-DAG:   %[[RETURN_VALUE:.*]] = call [64 x i8] @local_cabi_return_big_aggregate()
CHECK-DAG:   store [64 x i8] %[[RETURN_VALUE]], ptr %[[RETURN_ALLOCA]]
CHECK-DAG:   %[[RETURN_ALLOCA_INT_16:.*]] = add i64 %[[RETURN_ALLOCA_INT]], 16
CHECK-DAG:   %[[RETURN_ALLOCA_16:.*]] = inttoptr i64 %[[RETURN_ALLOCA_INT_16]] to ptr
CHECK-DAG:   %[[TO_RETURN:.*]] = load i64, ptr %[[RETURN_ALLOCA_16]]
CHECK: }

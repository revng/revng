;
; Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
;

CHECK: define i64 @local_raw_primitives_on_registers(i64 %[[ARG1:.*]], i64 %[[ARG2:.*]]) [[IGN:.*]] {
CHECK-DAG: add i64 [[IGN:.*]]%[[ARG1]]
CHECK-DAG: add i64 [[IGN:.*]]%[[ARG2]]
CHECK: }

CHECK: define i64 @local_raw_pointers_on_registers(i64 %[[ARG1:.*]], i64 %[[ARG2:.*]]) [[IGN:.*]] {
CHECK-DAG: %[[ARG1_PTR:.*]] = inttoptr i64 %[[ARG1]] to ptr
CHECK-DAG: load i64, ptr %[[ARG1_PTR:.*]]
CHECK-DAG: %[[ARG2_PTR:.*]] = inttoptr i64 %[[ARG2]] to ptr
CHECK-DAG: load i64, ptr %[[ARG2_PTR]]
CHECK: }

CHECK: define i64 @local_raw_primitives_on_stack(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG:.*]]) [[IGN:.*]] {
CHECK-DAG:   %[[STACK_ARG_AO:.*]] = call i64 @AddressOf([[IGN:.*]]i64 %[[STACK_ARG]])
CHECK-DAG:   %[[STACK_ARG8:.*]] = add i64 %[[STACK_ARG_AO]], 8
CHECK-DAG:   %[[STACK_ARG8_PTR:.*]] = inttoptr i64 %[[STACK_ARG8]] to ptr
CHECK-DAG:   load i64, ptr %[[STACK_ARG8_PTR:.*]]
CHECK-DAG:   %[[STACK_ARG_PTR:.*]] = inttoptr i64 %[[STACK_ARG_AO]] to ptr
CHECK-DAG:   load i64, ptr %[[STACK_ARG_PTR]]
CHECK: }

CHECK: define i64 @local_cabi_primitives_on_registers(i64 %[[ARG1:.*]], i64 %[[ARG2:.*]]) [[IGN:.*]] {
CHECK-DAG: add i64 [[IGN:.*]][[ARG1]]
CHECK-DAG: add i64 [[IGN:.*]][[ARG2]]
CHECK: }

CHECK: define i64 @local_cabi_primitives_on_stack(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG1:.*]], i64 %[[STACK_ARG2:.*]]) [[IGN:.*]] {
CHECK-DAG:   %[[IGN:.*]] = add i64 %[[IGN:.*]]%[[STACK_ARG1]]
CHECK-DAG:   %[[IGN:.*]] = add i64 %[[IGN:.*]]%[[STACK_ARG2]]
CHECK: }

CHECK: define i64 @local_cabi_aggregate_on_registers(i64 %[[ARG1:.*]]) [[IGN:.*]] {
CHECK-DAG:   %[[ARG1_AO:.*]] = call i64 @AddressOf([[IGN:.*]]i64 %[[ARG1]])
CHECK-DAG:   %[[FIELD1_PTR:.*]] = inttoptr i64 %[[ARG1_AO]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD1_PTR]]
CHECK-DAG:   %[[FIELD2_ADDR:.*]] = add i64 %[[ARG1_AO]], 8
CHECK-DAG:   %[[FIELD2_PTR:.*]] = inttoptr i64 %[[FIELD2_ADDR]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD2_PTR]], align 8
CHECK: }

CHECK: define i64 @local_cabi_aggregate_on_stack(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG:.*]]) [[IGN:.*]] {
CHECK-DAG:   %[[STACK_ARG_AO:.*]] = call i64 @AddressOf([[IGN:.*]]i64 %[[STACK_ARG]])
CHECK-DAG:   %[[FIELD1_PTR:.*]] = inttoptr i64 %[[STACK_ARG_AO]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD1_PTR]]
CHECK-DAG:   %[[FIELD2_ADDR:.*]] = add i64 %[[STACK_ARG_AO]], 8
CHECK-DAG:   %[[FIELD2_PTR:.*]] = inttoptr i64 %[[FIELD2_ADDR]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD2_PTR]]
CHECK: }

CHECK: define i64 @local_cabi_aggregate_on_stack_and_registers(i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[IGN:.*]], i64 %[[STACK_ARG:.*]]) [[IGN:.*]] {
CHECK-DAG:   %[[STACK_ARG_AO:.*]] = call i64 @AddressOf([[IGN:.*]]i64 %[[STACK_ARG]])
CHECK-DAG:   %[[FIELD1_PTR:.*]] = inttoptr i64 %[[STACK_ARG_AO]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD1_PTR]]
CHECK-DAG:   %[[FIELD2_ADDR:.*]] = add i64 %[[STACK_ARG_AO]], 8
CHECK-DAG:   %[[FIELD2_PTR:.*]] = inttoptr i64 %[[FIELD2_ADDR]] to ptr
CHECK-DAG:   load i64, ptr %[[FIELD2_PTR]]
CHECK: }

CHECK: define i64 @local_caller() [[IGN:.*]] {
CHECK-DAG:   = call i64 @local_raw_primitives_on_registers(i64 2, i64 1)
CHECK-DAG:   = call i64 @local_raw_pointers_on_registers(i64 %[[ARG:.*]], i64 %[[ARG]])
CHECK-DAG:   %[[STACK:.*]] = call i64 @revng_call_stack_arguments([[IGN:.*]], i64 16)
CHECK-DAG:   = call i64 @local_raw_primitives_on_stack(i64 4, i64 3, i64 2, i64 1, i64 5, i64 6, i64 %[[STACK]])
CHECK-DAG:   = call i64 @local_cabi_primitives_on_registers(i64 1, i64 2)
TODO: devise a pipeline that highlights both arguments as immediates
CHECK-DAG:   = call i64 @local_cabi_primitives_on_stack(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 [[SCALAR1:.*]], i64 [[SCALAR2:.*]])
CHECK-DAG:   %[[AGGREGATE:.*]] = call i64 @revng_call_stack_arguments([[IGN:.*]], i64 16)
CHECK-DAG:   = call i64 @local_cabi_aggregate_on_registers(i64 %[[AGGREGATE]])
CHECK-DAG:   %[[AGGREGATE:.*]] = call i64 @revng_call_stack_arguments([[IGN:.*]], i64 16)
CHECK-DAG:   = call i64 @local_cabi_aggregate_on_stack(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 %[[AGGREGATE]])
CHECK-DAG:   %[[AGGREGATE:.*]] = call i64 @revng_call_stack_arguments([[IGN:.*]], i64 16)
CHECK-DAG:   = call i64 @local_cabi_aggregate_on_stack_and_registers(i64 1, i64 2, i64 3, i64 4, i64 5, i64 %[[AGGREGATE]])
CHECK: }

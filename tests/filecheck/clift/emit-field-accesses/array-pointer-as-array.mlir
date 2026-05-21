//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!f1 = !clift.func<
  "1001" as "f1" : !void(!int32_t$ptr)
>

!f2 = !clift.func<
  "1002" as "f2" : !void(!int32_t$ptr, !generic64_t)
>

module attributes {clift.module} {

  // pointer as array access: *(p + 3) should become p[3], with a constant index

  clift.func @test_pointer_as_array__index<!f1>(%arg0 : !int32_t$ptr) {
    clift.expr {
      %0 = clift.bitcast %arg0 : !int32_t$ptr -> !generic64_t
      %1 = clift.imm 12 : !generic64_t
      %2 = clift.add %0, %1 : !generic64_t
      %3 = clift.bitcast %2 : !generic64_t -> !int32_t$ptr
      clift.yield %3 : !int32_t$ptr
    }
  }

  // CHECK: clift.func @test_pointer_as_array__index<!f1_>([[POINTER:%[a-z0-9]+]]: {{.*}})
  // CHECK: [[IMMEDIATE:%[0-9]+]] = clift.imm 3
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[POINTER]], [[IMMEDIATE]]
  // CHECK: [[ADDRESSOF:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF]] : !clift.ptr<8 to !int32_t>


  // pointer as array access: *(p + i) should become p[i], with the index passed
  // as an argument

  clift.func @test_pointer_as_array_variable_index<!f2>(%arg0 : !int32_t$ptr, %arg1 : !generic64_t) {
    clift.expr {
      %0 = clift.bitcast %arg0 : !int32_t$ptr -> !generic64_t
      %1 = clift.imm 4 : !generic64_t
      %2 = clift.mul %arg1, %1 : !generic64_t
      %3 = clift.add %0, %2 : !generic64_t
      %4 = clift.bitcast %3 : !generic64_t -> !int32_t$ptr
      clift.yield %4 : !int32_t$ptr
    }
  }

  // CHECK: clift.func @test_pointer_as_array_variable_index<!f2_>([[POINTER:%[a-z0-9]+]]: {{.*}}, [[INDEX:%[a-z0-9]+]]: {{.*}})
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[POINTER]], [[INDEX]]
  // CHECK: [[ADDRESSOF:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF]] : !clift.ptr<8 to !int32_t>


  // pointer as array access: *(p + i) should become p[i], with the index passed
  // as an argument, using `ptr_add`

  clift.func @test_pointer_as_array_ptr_add<!f2>(%arg0 : !int32_t$ptr, %arg1 : !generic64_t) {
    clift.expr {
      %0 = clift.ptr_add %arg0, %arg1 : (!int32_t$ptr, !generic64_t)
      clift.yield %0 : !int32_t$ptr
    }
  }

  // CHECK: clift.func @test_pointer_as_array_ptr_add<!f2_>([[POINTER:%[a-z0-9]+]]: {{[^,]*}}, [[INDEX:%[a-z0-9]+]]: {{.*}})
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[POINTER]], [[INDEX]]
  // CHECK: [[ADDRESSOF:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF]] : !clift.ptr<8 to !int32_t>
}

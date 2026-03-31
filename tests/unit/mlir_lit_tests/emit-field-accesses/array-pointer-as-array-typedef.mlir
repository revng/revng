//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s -emit-field-accesses -canonicalize 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!generic64_t = !clift.primitive<generic 8>
!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>
!int32_t$ptr$typedef = !clift.typedef<"/type-definition/100-TypedefDefinition" : !int32_t$ptr>

!f1 = !clift.func<
  "1001" as "f1" : !void(!int32_t$ptr$typedef, !generic64_t)
>

!f2 = !clift.func<
  "1002" as "f2" : !void(!int32_t$ptr$typedef)
>

module attributes {clift.module} {

  // pointer as array access: *(p + i) should become p[i], with the index passed
  // as an argument. The `ptr` is wrapped inside a `typedef`.

  clift.func @test_typedef_pointer_as_array<!f1>(%arg0 : !int32_t$ptr$typedef, %arg1 : !generic64_t) {
    clift.expr {
      %0 = clift.cast<bitcast> %arg0 : !int32_t$ptr$typedef -> !generic64_t
      %1 = clift.imm 4 : !generic64_t
      %2 = clift.mul %arg1, %1 : !generic64_t
      %3 = clift.add %0, %2 : !generic64_t
      %4 = clift.cast<bitcast> %3 : !generic64_t -> !int32_t$ptr
      clift.yield %4 : !int32_t$ptr
    }
  }

  // CHECK: clift.func @test_typedef_pointer_as_array<!f1_>([[POINTER:%[a-z0-9]+]]: {{.*}}, [[INDEX:%[a-z0-9]+]]: {{.*}})
  // CHECK: [[CAST:%[0-9]+]] = clift.cast<bitcast> [[POINTER]] : !_type_definition_100_TypedefDefinition -> !clift.ptr<8 to !int32_t>
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[CAST]], [[INDEX]]
  // CHECK: [[ADDRESSOF:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF]] : !clift.ptr<8 to !int32_t>


  // pointer as array access: *(p + 3) should become p[3], with a constant
  // index. The `ptr` is wrapped inside a `typedef`.
  clift.func @test_typedef_constant_index<!f2>(%arg0 : !int32_t$ptr$typedef) {
    clift.expr {
      %0 = clift.cast<bitcast> %arg0 : !int32_t$ptr$typedef -> !generic64_t
      %1 = clift.imm 12 : !generic64_t
      %2 = clift.add %0, %1 : !generic64_t
      %3 = clift.cast<bitcast> %2 : !generic64_t -> !int32_t$ptr
      clift.yield %3 : !int32_t$ptr
    }
  }

  // CHECK: clift.func @test_typedef_constant_index<!f2_>([[POINTER:%[a-z0-9]+]]: {{.*}})
  // CHECK: [[IMM:%[0-9]+]] = clift.imm 3
  // CHECK: [[CAST:%[0-9]+]] = clift.cast<bitcast> [[POINTER]] : !_type_definition_100_TypedefDefinition -> !clift.ptr<8 to !int32_t>
  // CHECK: [[SUBSCRIPT:%[0-9]+]] = clift.subscript [[CAST]], [[IMM]]
  // CHECK: [[ADDRESSOF:%[0-9]+]] = clift.addressof [[SUBSCRIPT]]
  // CHECK: clift.yield [[ADDRESSOF]] : !clift.ptr<8 to !int32_t>
}

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --emit-field-accesses --canonicalize | FileCheck %s
// RUN: %root/bin/revng clift-opt %s --refine-types-and-accesses -o /dev/null

!void = !clift.void
!generic64_t = !clift.int<generic 8>
!int32_t = !clift.int<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!inner = !clift.struct<"1" : size(4) {
  "" : offset(0) !int32_t
}>
!array = !clift.array<2 x !int32_t>
!outer = !clift.struct<"2" : size(16) {
  "" : offset(4) !inner,
  "" : offset(8) !array
}>
!outer$const = !clift.const<!outer>
!outer$ptr = !clift.ptr<8 to !outer$const>
!f = !clift.func<"1000" : !void(!outer$ptr)>

module attributes {clift.module} {
  // Both the indirect outer access and the direct inner access inherit const.
  // The original pointer cast still permits writing through a non-const pointer.
  clift.func @nested_store<!f>(%p : !outer$ptr) {
    clift.expr {
      %address = clift.bitcast %p : !outer$ptr -> !generic64_t
      %offset = clift.imm 4 : !generic64_t
      %sum = clift.add %address, %offset : !generic64_t
      %pointer = clift.bitcast %sum : !generic64_t -> !int32_t$ptr
      %field = clift.indirection %pointer : !int32_t$ptr
      %value = clift.imm 2 : !int32_t
      %result = clift.assign %field, %value : !int32_t
      clift.yield %result : !int32_t
    }
  }

  // CHECK-LABEL: clift.func @nested_store
  // CHECK: [[OUTER:%[0-9]+]] = clift.ptr_access<0> %arg0
  // CHECK-SAME: -> !clift.const<!_1_>
  // CHECK: [[INNER:%[0-9]+]] = clift.access<0> [[OUTER]]
  // CHECK-SAME: -> !clift.const<!int32_t>
  // CHECK: [[ADDRESS:%[0-9]+]] = clift.addressof [[INNER]]
  // CHECK-SAME: : !clift.ptr<8 to !clift.const<!int32_t>>
  // CHECK: [[CAST:%[0-9]+]] = clift.bitcast [[ADDRESS]]
  // CHECK-SAME: -> !clift.ptr<8 to !int32_t>
  // CHECK: [[FIELD:%[0-9]+]] = clift.indirection [[CAST]]
  // CHECK: clift.assign [[FIELD]], {{%[0-9]+}} : !int32_t

  // Array decay must preserve the qualifier inherited from the containing struct.
  clift.func @array_element<!f>(%p : !outer$ptr) {
    clift.expr {
      %address = clift.bitcast %p : !outer$ptr -> !generic64_t
      %offset = clift.imm 12 : !generic64_t
      %sum = clift.add %address, %offset : !generic64_t
      %pointer = clift.bitcast %sum : !generic64_t -> !int32_t$ptr
      clift.yield %pointer : !int32_t$ptr
    }
  }

  // CHECK-LABEL: clift.func @array_element
  // CHECK: [[ARRAY:%[0-9]+]] = clift.ptr_access<1> %arg0
  // CHECK-SAME: -> !clift.const<!clift.array<2 x !int32_t>>
  // CHECK: [[DECAY:%[0-9]+]] = clift.decay [[ARRAY]]
  // CHECK-SAME: -> !clift.ptr<8 to !clift.const<!int32_t>>
  // CHECK: [[ELEMENT:%[0-9]+]] = clift.subscript [[DECAY]], {{%[0-9]+}}
  // CHECK: [[ADDRESS:%[0-9]+]] = clift.addressof [[ELEMENT]]
  // CHECK-SAME: : !clift.ptr<8 to !clift.const<!int32_t>>
  // CHECK: [[CAST:%[0-9]+]] = clift.bitcast [[ADDRESS]]
  // CHECK-SAME: -> !clift.ptr<8 to !int32_t>
  // CHECK: clift.yield [[CAST]] : !clift.ptr<8 to !int32_t>
}

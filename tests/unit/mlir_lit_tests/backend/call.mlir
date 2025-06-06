//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1002-CABIFunctionDefinition" : !int32_t(!int32_t, !int32_t)
>

module attributes {clift.module} {
  // CHECK: int32_t fun_0x40001002(int32_t x, int32_t y) {
  clift.func @f<!f>(%arg0 : !int32_t, %arg1 : !int32_t) attributes {
    handle = "/function/0x40001002:Code_x86_64"
  } {
    // CHECK: return
    clift.return {
      %f = clift.use @f : !f

      %comma = clift.comma %arg1, %arg1 : !int32_t, !int32_t

      // CHECK: fun_0x40001002(x, (y, y))
      %result = clift.call %f(%arg0, %comma) : !f

      clift.yield %result : !int32_t
    }
    // CHECK: ;
  }
  // CHECK: }
}

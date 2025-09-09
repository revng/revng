//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe model-verify-clift %S/model.yml %s /dev/null

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!s = !clift.struct<"/type-definition/1001-StructDefinition" : size(1) {}>
!u = !clift.union<"/type-definition/1002-UnionDefinition" : { !int32_t }>
!e = !clift.enum<"/type-definition/1003-EnumDefinition" : !int32_t { 0 }>
!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>
!r = !clift.func<"/type-definition/1005-RawFunctionDefinition" : !void()>
!t = !clift.typedef<"/type-definition/1006-TypedefDefinition" : !int32_t>

module attributes {clift.module} {
  clift.func @helper<!f>() attributes { handle = "/helper-function/foo" }
  clift.func @imported<!f>() attributes { handle = "/dynamic-function/foo" }
  clift.func @isolated<!f>() attributes { handle = "/function/0x40001001:Code_x86_64" } {
    %1 = clift.local : !s
    %2 = clift.local : !u
    %3 = clift.local : !e
    %4 = clift.local : !clift.ptr<8 to !f>
    %5 = clift.local : !clift.ptr<8 to !r>
    %6 = clift.local : !t
  }
}

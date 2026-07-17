//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng pipe import-descriptive-info %S/../model.yml %s /dev/stdout | %root/bin/revng clift-opt | FileCheck %s

!uint64_t = !clift.int<unsigned 8>
!int64_t = !clift.int<signed 8>

// The size_t typedef matches the `size_t` PrimitiveAlias entry in the
// SystemV_x86_64 ABI (Unsigned, 8 bytes), so the name is preserved.

// CHECK: !size_t = !clift.typedef<
// CHECK:   "/type-definition/3001-TypedefDefinition" as "size_t" : !uint64_t
// CHECK: >
!size_t = !clift.typedef<
  "/type-definition/3001-TypedefDefinition" as "whatever" : !uint64_t
>

// The ptrdiff_t typedef declares the WRONG kind (Unsigned) and does not
// match the `ptrdiff_t` PrimitiveAlias entry (which is Signed, 8 bytes).
// Because of this, `isKnownPrimitiveAlias` returns false and the typedef
// falls back to a generic auto-generated name.

// CHECK: !typedef_3002_ = !clift.typedef<
// CHECK:   "/type-definition/3002-TypedefDefinition" as "typedef_3002" : !int64_t
// CHECK: >
!ptrdiff_t = !clift.typedef<
  "/type-definition/3002-TypedefDefinition" as "whatever" : !int64_t
>

module attributes {clift.module, clift.types = [!size_t, !ptrdiff_t]} {
}

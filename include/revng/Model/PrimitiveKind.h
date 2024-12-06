#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

/* TUPLE-TREE-YAML
name: PrimitiveKind
type: enum
members:
  - name: Void
    doc: The `void` type.
  - name: Generic
    doc: |-
      The most generic primitive kind: it can be any of the other primitive
      kinds, except for `Void`.
  - name: PointerOrNumber
    doc: |-
      A kind representing either a `Number` kind or a pointer.
      This can also be seen as not-a-`Float`.
  - name: Number
    doc: |-
      A two's complement integer number, either `Signed` or `Unsigned`.
  - name: Unsigned
    doc: |-
      An `unsigned` two's complement integer number.
  - name: Signed
    doc: |-
      An `signed` two's complement integer number.
  - name: Float
    doc: |-
      A IEEE 754 floating point number.
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/PrimitiveKind.h"

namespace model::PrimitiveKind {

constexpr llvm::StringRef getCPrefix(Values V) {
  switch (V) {
  case PrimitiveKind::Void:
    return "void";
  case PrimitiveKind::Unsigned:
    return "uint";
  case PrimitiveKind::Signed:
    return "int";
  case PrimitiveKind::Float:
    return "float";
  case PrimitiveKind::Generic:
    return "generic";
  case PrimitiveKind::Number:
    return "number";
  case PrimitiveKind::PointerOrNumber:
    return "pointer_or_number";
  default:
    revng_abort("Unsupported primitive kind.");
  }
}

constexpr std::pair<Values, llvm::StringRef>
tryConsumeCPrefix(llvm::StringRef Name) {
  // Parse the prefix for the kind
  if (Name.consume_front("generic"))
    return { PrimitiveKind::Generic, Name };
  else if (Name.consume_front("uint"))
    return { PrimitiveKind::Unsigned, Name };
  else if (Name.consume_front("number"))
    return { PrimitiveKind::Number, Name };
  else if (Name.consume_front("pointer_or_number"))
    return { PrimitiveKind::PointerOrNumber, Name };
  else if (Name.consume_front("int"))
    return { PrimitiveKind::Signed, Name };
  else if (Name.consume_front("float"))
    return { PrimitiveKind::Float, Name };
  else if (Name.consume_front("void"))
    return { PrimitiveKind::Void, Name };
  else
    return { PrimitiveKind::Invalid, Name };
}

} // namespace model::PrimitiveKind

#include "revng/Model/Generated/Late/PrimitiveKind.h"

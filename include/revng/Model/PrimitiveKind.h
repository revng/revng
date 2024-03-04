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
  - name: Generic
  - name: PointerOrNumber
  - name: Number
  - name: Unsigned
  - name: Signed
  - name: Float
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
  else
    return { PrimitiveKind::Invalid, Name };
}

} // namespace model::PrimitiveKind

#include "revng/Model/Generated/Late/PrimitiveKind.h"

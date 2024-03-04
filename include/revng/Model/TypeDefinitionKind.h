#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Generated/Early/TypeDefinitionKind.h"

namespace model::TypeDefinitionKind {

constexpr llvm::StringRef automaticNamePrefix(Values V) {
  switch (V) {
  case StructDefinition:
    return "struct_";
  case UnionDefinition:
    return "union_";
  case EnumDefinition:
    return "enum_";
  case TypedefDefinition:
    return "typedef_";
  case CABIFunctionDefinition:
    return "cabifunction_";
  case RawFunctionDefinition:
    return "rawfunction_";
  default:
    revng_abort("Unsupported type definition "
                "kind.");
  }
}

} // namespace model::TypeDefinitionKind

#include "revng/Model/Generated/Late/TypeDefinitionKind.h"

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

inline constexpr llvm::StringRef consumeNamePrefix(llvm::StringRef String) {
  // Skip `Invalid`: we never want to emit names for invalid types,
  // so parsing them doesn't make much sense either.
  auto Index = static_cast<Values>(static_cast<uint64_t>(Values::Invalid) + 1);

  while (Index < Values::Count) {
    llvm::StringRef Serialized = automaticNamePrefix(Index);
    if (String.starts_with(Serialized))
      return String.substr(Serialized.size());

    Index = static_cast<Values>(static_cast<uint64_t>(Index) + 1);
  }

  return String;
}

} // namespace model::TypeDefinitionKind

#include "revng/Model/Generated/Late/TypeDefinitionKind.h"

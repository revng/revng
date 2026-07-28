#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinitionKind.h"
#include "revng/Support/YAMLTraits.h"

namespace model {

/// Resolve a type-definition name (as printed by the C emitter, either a chosen
/// name like `my_struct` or the automatic `struct_50`) to a model type.
///
/// The name is first matched against the name of each definition. Failing that,
/// the automatic `<prefix>_<id>` form is parsed: when `Kind` is set only
/// definitions of that kind are considered and the id is read after the last
/// `_`; otherwise the kind is taken from the automatic prefix (`struct_`,
/// `enum_`, ...). Returns an empty type if nothing matches.
inline model::UpcastableType
getTypeDefinitionByNameOrID(const model::Binary &Binary,
                            llvm::StringRef Name,
                            std::optional<model::TypeDefinitionKind::Values>
                              Kind = std::nullopt) {

  for (const model::UpcastableTypeDefinition &Definition :
       Binary.TypeDefinitions())
    if ((not Kind or Definition->Kind() == *Kind)
        and Definition->Name() == Name)
      return Binary.makeType(Definition->key());

  // The id part of an automatic name must be a plain sequence of digits;
  // otherwise the `<prefix>_<id>` form does not apply.
  auto ByKindAndID =
    [&Binary](model::TypeDefinitionKind::Values ResolvedKind,
              llvm::StringRef IDText) -> model::UpcastableType {
    bool AllDigits = not IDText.empty()
                     and llvm::all_of(IDText, [](char Character) {
                           return Character >= '0' and Character <= '9';
                         });
    if (not AllDigits)
      return model::UpcastableType::empty();

    if (auto MaybeID = fromString<uint64_t>(IDText))
      return Binary.makeType(model::TypeDefinition::Key{ *MaybeID,
                                                         ResolvedKind });
    else
      llvm::consumeError(MaybeID.takeError());
    return model::UpcastableType::empty();
  };

  if (Kind) {
    size_t Tail = Name.rfind('_');
    if (Tail != llvm::StringRef::npos)
      return ByKindAndID(*Kind, Name.substr(Tail + 1));
  } else {
    namespace Kinds = model::TypeDefinitionKind;
    for (auto Value = static_cast<Kinds::Values>(Kinds::Invalid + 1);
         Value < Kinds::Count;
         Value = static_cast<Kinds::Values>(Value + 1)) {
      llvm::StringRef Prefix = Kinds::automaticNamePrefix(Value);
      if (Name.starts_with(Prefix))
        return ByKindAndID(Value, Name.substr(Prefix.size()));
    }
  }

  return model::UpcastableType::empty();
}

} // namespace model

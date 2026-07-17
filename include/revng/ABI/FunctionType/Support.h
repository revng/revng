#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/Support/MathExtras.h"

#include "revng/ADT/Concepts.h"
#include "revng/ADT/STLExtras.h"
#include "revng/Model/Binary.h"
#include "revng/Model/DefinedType.h"
#include "revng/Support/AlignmentHelpers.h"

namespace abi::FunctionType {

/// Replace all the references to a type definition with \p OldKey key with
/// the \p NewType. It also erases the old type definition.
///
/// \param Old The type references of which should be replaced.
/// \param New The new type to replace references to.
inline void replaceTypeDefinition(const model::TypeDefinition::Key &Old,
                                  const model::TypeDefinitionReference &New,
                                  TupleTree<model::Binary> &Binary) {
  using Reference = model::TypeDefinitionReference;
  Binary.replaceReferencesIf(New, [&Old](const Reference &Path) -> bool {
    if (Path.empty())
      return false;

    return Old == Path.getConst()->key();
  });
}
inline void replaceTypeDefinition(const model::TypeDefinition::Key &Old,
                                  const model::DefinedType &New,
                                  TupleTree<model::Binary> &Binary) {
  return replaceTypeDefinition(Old, New.Definition(), Binary);
}
inline void replaceTypeDefinition(const model::TypeDefinition::Key &O,
                                  const model::Type &N,
                                  TupleTree<model::Binary> &B) {
  return replaceTypeDefinition(O, llvm::cast<model::DefinedType>(N), B);
}

/// Filters a list of upcastable types.
///
/// \tparam DerivedType The desired type to filter based on
/// \param Types The list of types to filter
/// \return filtered list
template<std::derived_from<model::TypeDefinition> DerivedType,
         RangeOf<model::UpcastableTypeDefinition> OwningRange,
         RangeOf<model::TypeDefinition *> ViewRange =
           std::vector<model::TypeDefinition *>>
std::vector<DerivedType *>
filterTypes(OwningRange &FilterFrom, const ViewRange &Ignored = {}) {
  std::vector<DerivedType *> Result;
  for (model::UpcastableTypeDefinition &Type : FilterFrom)
    if (Type && !llvm::is_contained(Ignored, &*Type))
      if (auto *Cast = llvm::dyn_cast<DerivedType>(Type.get()))
        Result.emplace_back(Cast);
  return Result;
}

} // namespace abi::FunctionType

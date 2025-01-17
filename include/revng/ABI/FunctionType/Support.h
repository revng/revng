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

namespace abi::FunctionType {

/// Replace all the references to a type definition with \p OldKey key with
/// the \p NewType. It also erases the old type definition.
///
/// \param Old The type references of which should be replaced.
/// \param New The new type to replace references to.
inline void replaceTypeDefinition(const model::TypeDefinition::Key &Old,
                                  const model::DefinitionReference &New,
                                  TupleTree<model::Binary> &Binary) {
  using Reference = model::DefinitionReference;
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

/// Takes care of extending (padding) the size of a stack argument.
///
/// \note This only accounts for the post-padding (extension).
///       Pre-padding (offset) needs to be taken care of separately.
///
/// \param RealSize The size of the argument without the padding.
/// \param RegisterSize The size of a register under the given architecture.
///
/// \return The size of the argument with the padding.
inline constexpr uint64_t paddedSizeOnStack(uint64_t RealSize,
                                            uint64_t RegisterSize) {
  revng_assert(llvm::isPowerOf2_64(RegisterSize));
  revng_assert(RealSize != 0, "0-sized stack entries are not supported.");

  if (RealSize <= RegisterSize)
    return RegisterSize;

  RealSize += RegisterSize - 1;
  RealSize &= ~(RegisterSize - 1);

  return RealSize;
}

/// Filters a list of upcastable types.
///
/// \tparam DerivedType The desired type to filter based on
/// \param Types The list of types to filter
/// \return filtered list
template<derived_from<model::TypeDefinition> DerivedType,
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

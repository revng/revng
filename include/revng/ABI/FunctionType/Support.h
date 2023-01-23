#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/Support/MathExtras.h"

#include "revng/ADT/Concepts.h"
#include "revng/ADT/STLExtras.h"
#include "revng/Model/Binary.h"

namespace abi::FunctionType {

/// Replace all the references to `OldKey` with the references to
/// the newly added `NewType`. It also erases the old type.
///
/// \param OldKey The type references of which should be replaced.
/// \param NewTypePath The reference to a type references should be replaced to.
/// \param Model The tuple tree where replacement should take place in.
///
/// \return The new path to the added type.
const model::TypePath &replaceAllUsesWith(const model::Type::Key &OldKey,
                                          const model::TypePath &NewTypePath,
                                          TupleTree<model::Binary> &Model);

/// Takes care of extending (padding) the size of a stack argument.
///
/// \note This only accounts for the post-padding (extension).
///       Pre-padding (offset) needs to be taken care of separately.
///
/// \param RealSize The size of the argument without the padding.
/// \param RegisterSize The size of a register under the given architecture.
///
/// \return The size of the argument with the padding.
inline constexpr uint64_t
paddedSizeOnStack(uint64_t RealSize, uint64_t RegisterSize) {
  revng_assert(llvm::isPowerOf2_64(RegisterSize));

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
template<derived_from<model::Type> DerivedType>
std::vector<DerivedType *>
filterTypes(SortedVector<UpcastablePointer<model::Type>> &Types) {
  std::vector<DerivedType *> Result;
  for (model::UpcastableType &Type : Types)
    if (auto *Upscaled = llvm::dyn_cast<DerivedType>(Type.get()))
      Result.emplace_back(Upscaled);
  return Result;
}

} // namespace abi::FunctionType

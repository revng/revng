#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/MathExtras.h"

#include "revng/Model/Binary.h"

namespace abi::FunctionType {

constexpr inline model::PrimitiveTypeKind::Values
selectTypeKind(model::Register::Values) {
  // TODO: implement a way to determine the register type. At the very least
  // we should be able to differentiate GPRs from the vector registers.

  return model::PrimitiveTypeKind::PointerOrNumber;
}

inline model::QualifiedType
buildType(model::Register::Values Register, model::Binary &Binary) {
  model::PrimitiveTypeKind::Values Kind = selectTypeKind(Register);
  size_t Size = model::Register::getSize(Register);
  return model::QualifiedType(Binary.getPrimitiveType(Kind, Size), {});
}

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

} // namespace abi::FunctionType

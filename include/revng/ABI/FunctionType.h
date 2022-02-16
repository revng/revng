#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Types.h"

namespace abi::FunctionType {

/// Best effort `CABIFunctionType` to `RawFunctionType` conversion.
///
/// If `ABI` is not specified, `TheBinaryToAddSupportTypesTo.DefaultABI`
/// is used instead.
std::optional<model::CABIFunctionType>
tryConvertToCABI(const model::RawFunctionType &Function,
                 model::Binary &TheBinaryToAddSupportTypesTo,
                 std::optional<model::ABI::Values> ABI = std::nullopt);

/// Best effort `RawFunctionType` to `CABIFunctionType` conversion.
///
/// \note: this convertion is loosy since there's no way to represent some types
/// in `RawFunctionType` in a reversible manner.
std::optional<model::RawFunctionType>
tryConvertToRaw(const model::CABIFunctionType &Function,
                model::Binary &TheBinaryToAddSupportTypesTo);

/// Indicates the layout of arguments and return values of a function.
///
/// \note Such a layout is immutable.
struct Layout {
public:
  struct ReturnValueRegisters {
    llvm::SmallVector<model::Register::Values, 2> Registers;
  };

  struct Argument : public ReturnValueRegisters {
  public:
    struct StackSpan {
      uint64_t Offset;
      uint64_t Size;
    };

  public:
    std::optional<StackSpan> Stack;
  };

public:
  /// Extracts the information about argument and return value location layout
  /// from the \param Function.
  inline static Layout make(const model::TypePath &Function) {
    revng_assert(Function.isValid());
    if (auto *CABI = llvm::dyn_cast<model::CABIFunctionType>(Function.get()))
      return Layout(*CABI);
    else if (auto *Raw = llvm::dyn_cast<model::RawFunctionType>(Function.get()))
      return Layout(*Raw);
    else
      revng_abort("Layouts of non-function types are not supported.");
  }

  explicit Layout(const model::RawFunctionType &Function);
  explicit Layout(const model::CABIFunctionType &Function);

  bool verify() const;
  size_t argumentRegisterCount() const;
  size_t returnValueRegisterCount() const;

private:
  Layout() {}

public:
  llvm::SmallVector<Argument, 4> Arguments;
  ReturnValueRegisters ReturnValue;
  llvm::SmallVector<model::Register::Values, 24> CalleeSavedRegisters;
  uint64_t FinalStackOffset;
};

} // namespace abi::FunctionType

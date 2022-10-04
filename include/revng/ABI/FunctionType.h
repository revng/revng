#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Type.h"
#include "revng/Model/Types.h"
#include "revng/Support/Generator.h"

namespace abi::FunctionType {

namespace ArgumentKind {

enum Values {
  Scalar = 0,
  ReferenceToAggregate,
  ShadowPointerToAggregateReturnValue,
  Invalid,
};

inline const char *getName(Values Kind) {
  switch (Kind) {
  case Scalar:
    return "Scalar";
  case ReferenceToAggregate:
    return "ReferenceToAggregate";
  case ShadowPointerToAggregateReturnValue:
    return "ShadowPointerToAggregateReturnValue";
  default:;
  }
  return "Invalid";
}

} // end namespace ArgumentKind

/// Best effort `CABIFunctionType` to `RawFunctionType` conversion.
///
/// If `ABI` is not specified, `TheBinary.DefaultABI` is used instead.
std::optional<model::TypePath>
tryConvertToCABI(const model::RawFunctionType &Function,
                 TupleTree<model::Binary> &TheBinary,
                 std::optional<model::ABI::Values> ABI = std::nullopt);

/// Best effort `RawFunctionType` to `CABIFunctionType` conversion.
///
/// \note: this conversion is lossy since there's no way to represent some types
///        in `RawFunctionType` in a reversible manner.
model::TypePath convertToRaw(const model::CABIFunctionType &Function,
                             TupleTree<model::Binary> &TheBinary);

/// Indicates the layout of arguments and return values of a function.
struct Layout {
public:
  struct ReturnValue {
    model::QualifiedType Type;
    llvm::SmallVector<model::Register::Values, 2> Registers;
  };

  struct Argument : public ReturnValue {
  public:
    struct StackSpan {
      uint64_t Offset;
      uint64_t Size;

      StackSpan operator+(uint64_t Offset) const {
        return { this->Offset + Offset, Size };
      }
    };

  public:
    std::optional<StackSpan> Stack;
    ArgumentKind::Values Kind;
  };

public:
  llvm::SmallVector<Argument, 4> Arguments;
  llvm::SmallVector<ReturnValue, 2> ReturnValues;
  llvm::SmallVector<model::Register::Values, 24> CalleeSavedRegisters;
  uint64_t FinalStackOffset;

public:
  Layout() = default;

public:
  explicit Layout(const model::CABIFunctionType &Function);
  explicit Layout(const model::RawFunctionType &Function);

  /// Extracts the information about argument and return value location layout
  /// from the \param Function.
  static Layout make(const model::TypePath &Function) {
    revng_assert(Function.isValid());
    return make(*Function.get());
  }

  static Layout make(const model::Type &Function) {
    if (auto CABI = llvm::dyn_cast<model::CABIFunctionType>(&Function))
      return Layout(*CABI);
    else if (auto *Raw = llvm::dyn_cast<model::RawFunctionType>(&Function))
      return Layout(*Raw);
    else
      revng_abort("Layouts of non-function types are not supported.");
  }

public:
  bool verify() const debug_function;

  size_t argumentRegisterCount() const;
  size_t returnValueRegisterCount() const;
  llvm::SmallVector<model::Register::Values, 8> argumentRegisters() const;
  llvm::SmallVector<model::Register::Values, 8> returnValueRegisters() const;

  bool returnsAggregateType() const {
    using namespace abi::FunctionType::ArgumentKind;
    auto SPTAR = ShadowPointerToAggregateReturnValue;
    return (Arguments.size() >= 1 and Arguments[0].Kind == SPTAR);
  }

public:
  void dump() const debug_function {
    // TODO: accept an arbitrary stream

    //
    // Arguments
    //
    dbg << "Arguments:\n";
    for (const Argument &A : Arguments) {
      dbg << "  - Type: ";
      A.Type.dump();
      dbg << "\n";
      dbg << "    Registers: [";
      for (model::Register::Values Register : A.Registers)
        dbg << " " << model::Register::getName(Register).str();
      dbg << " ]\n";
      dbg << "    StackSpan: ";
      if (A.Stack) {
        dbg << "{ Offset: " << A.Stack->Offset << ", Size: " << A.Stack->Size
            << " }\n";
      } else {
        dbg << "no\n";
      }
      dbg << "    Kind: " << getName(A.Kind);
      dbg << "\n";
    }

    //
    // ReturnValues
    //
    dbg << "ReturnValues: \n";
    for (const ReturnValue &RV : ReturnValues) {
      dbg << "  - Type: ";
      RV.Type.dump();
      dbg << "\n";
      dbg << "    Registers: [";
      for (model::Register::Values Register : RV.Registers)
        dbg << " " << model::Register::getName(Register).str();
      dbg << " ]\n";
    }

    //
    // CalleeSavedRegisters
    //
    dbg << "CalleeSavedRegisters: [";
    for (model::Register::Values Register : CalleeSavedRegisters)
      dbg << " " << model::Register::getName(Register).str();
    dbg << " ]\n";

    //
    // FinalStackOffset
    //
    dbg << "FinalStackOffset: " << FinalStackOffset << "\n";
  }
};

} // namespace abi::FunctionType

#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"

#include "revng/ABI/Definition.h"
#include "revng/Model/Binary.h"
#include "revng/Model/QualifiedType.h"

namespace abi::FunctionType {

/// Best effort `CABIFunctionDefinition` to `RawFunctionDefinition` conversion.
///
/// \note: this conversion is lossy since there's no way to represent some types
///        in `RawFunctionDefinition` in a reversible manner.
model::DefinitionReference
convertToRaw(const model::CABIFunctionDefinition &Prototype,
             TupleTree<model::Binary> &TheBinary);

namespace ArgumentKind {

enum Values {
  Scalar,
  PointerToCopy,
  ReferenceToAggregate,
  ShadowPointerToAggregateReturnValue,

  Count
};

inline llvm::StringRef getName(Values Kind) {
  switch (Kind) {
  case Scalar:
    return "Scalar";
  case PointerToCopy:
    return "PointerToCopy";
  case ReferenceToAggregate:
    return "ReferenceToAggregate";
  case ShadowPointerToAggregateReturnValue:
    return "ShadowPointerToAggregateReturnValue";
  default:
    revng_abort("Unknown enum entry");
  }
}

inline Values fromName(llvm::StringRef Kind) {
  if (Kind == "Scalar")
    return Scalar;
  else if (Kind == "PointerToCopy")
    return PointerToCopy;
  else if (Kind == "ReferenceToAggregate")
    return ReferenceToAggregate;
  else if (Kind == "ShadowPointerToAggregateReturnValue")
    return ShadowPointerToAggregateReturnValue;
  else
    revng_abort("Unknown enum entry");
}

} // namespace ArgumentKind

namespace ReturnMethod {
enum Values {
  Void,
  ModelAggregate,
  Scalar,
  RegisterSet
};
} // namespace ReturnMethod

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
  explicit Layout(const model::CABIFunctionDefinition &Prototype);
  explicit Layout(const model::RawFunctionDefinition &Prototype);

  /// Extracts the information about argument and return value location layout
  /// from the \param Prototype.
  static Layout make(const model::DefinitionReference &Prototype) {
    revng_assert(Prototype.isValid());
    return make(*Prototype.getConst());
  }

  static Layout make(const model::TypeDefinition &Prototype) {
    if (auto CABI = llvm::dyn_cast<model::CABIFunctionDefinition>(&Prototype))
      return Layout(*CABI);
    else if (auto *R = llvm::dyn_cast<model::RawFunctionDefinition>(&Prototype))
      return Layout(*R);
    else
      revng_abort("Layouts of non-function types are not supported.");
  }

public:
  bool verify() const debug_function;

  size_t argumentRegisterCount() const;
  size_t returnValueRegisterCount() const;
  llvm::SmallVector<model::Register::Values, 8> argumentRegisters() const;
  llvm::SmallVector<model::Register::Values, 8> returnValueRegisters() const;

  bool hasSPTAR() const {
    using namespace abi::FunctionType::ArgumentKind;
    auto SPTAR = ShadowPointerToAggregateReturnValue;
    return (Arguments.size() >= 1 and Arguments[0].Kind == SPTAR);
  }

  ReturnMethod::Values returnMethod() const {
    if (hasSPTAR()) {
      revng_assert(ReturnValues.size() <= 1);
      revng_assert(Arguments.size() >= 1);
      return ReturnMethod::ModelAggregate;
    } else if (ReturnValues.size() == 0) {
      return ReturnMethod::Void;
    } else if (ReturnValues.size() > 1) {
      return ReturnMethod::RegisterSet;
    } else if (not ReturnValues[0].Type.isScalar()) {
      revng_assert(ReturnValues.size() == 1);
      return ReturnMethod::ModelAggregate;
    } else {
      revng_assert(ReturnValues.size() == 1);
      revng_assert(ReturnValues[0].Type.isScalar());
      return ReturnMethod::Scalar;
    }
  }

  // \note Call only if returnMethod() is ModelAggregate
  const model::QualifiedType returnValueAggregateType() const {
    revng_assert(returnMethod() == ReturnMethod::ModelAggregate);

    if (hasSPTAR()) {
      revng_assert(ReturnValues.size() <= 1);
      revng_assert(Arguments.size() >= 1);
      return Arguments[0].Type.stripPointer();
    } else {
      revng_assert(ReturnValues.size() == 1);
      revng_assert(!ReturnValues[0].Type.isScalar());
      return ReturnValues[0].Type;
    }
  }

public:
  void dump() const debug_function;
};

inline std::span<const model::Register::Values>
calleeSavedRegisters(const model::CABIFunctionDefinition &Prototype) {
  return abi::Definition::get(Prototype.ABI()).CalleeSavedRegisters();
}

inline std::span<const model::Register::Values>
calleeSavedRegisters(const model::RawFunctionDefinition &Prototype) {
  return Prototype.PreservedRegisters();
}

inline std::span<const model::Register::Values>
calleeSavedRegisters(const model::TypeDefinition &Prototype) {
  if (auto CABI = llvm::dyn_cast<model::CABIFunctionDefinition>(&Prototype))
    return calleeSavedRegisters(*CABI);
  else if (auto *Raw = llvm::dyn_cast<model::RawFunctionDefinition>(&Prototype))
    return calleeSavedRegisters(*Raw);
  else
    revng_abort("Layouts of non-function types are not supported.");
}

inline std::span<const model::Register::Values>
calleeSavedRegisters(const model::DefinitionReference &Prototype) {
  revng_assert(Prototype.isValid());
  return calleeSavedRegisters(*Prototype.getConst());
}

uint64_t finalStackOffset(const model::CABIFunctionDefinition &Prototype);
inline uint64_t
finalStackOffset(const model::RawFunctionDefinition &Prototype) {
  return Prototype.FinalStackOffset();
}

inline uint64_t finalStackOffset(const model::TypeDefinition &Prototype) {
  if (auto CABI = llvm::dyn_cast<model::CABIFunctionDefinition>(&Prototype))
    return finalStackOffset(*CABI);
  else if (auto *Raw = llvm::dyn_cast<model::RawFunctionDefinition>(&Prototype))
    return finalStackOffset(*Raw);
  else
    revng_abort("Layouts of non-function types are not supported.");
}

inline uint64_t finalStackOffset(const model::DefinitionReference &Prototype) {
  revng_assert(Prototype.isValid());
  return finalStackOffset(*Prototype.getConst());
}

struct UsedRegisters {
  llvm::SmallVector<model::Register::Values, 8> Arguments;
  llvm::SmallVector<model::Register::Values, 8> ReturnValues;
};
UsedRegisters usedRegisters(const model::CABIFunctionDefinition &Prototype);

inline UsedRegisters
usedRegisters(const model::RawFunctionDefinition &Prototype) {
  UsedRegisters Result;
  for (const model::NamedTypedRegister &Register : Prototype.Arguments())
    Result.Arguments.emplace_back(Register.Location());
  for (const model::NamedTypedRegister &Register : Prototype.ReturnValues())
    Result.ReturnValues.emplace_back(Register.Location());
  return Result;
}

inline UsedRegisters usedRegisters(const model::TypeDefinition &Prototype) {
  if (auto CABI = llvm::dyn_cast<model::CABIFunctionDefinition>(&Prototype))
    return usedRegisters(*CABI);
  else if (auto *Raw = llvm::dyn_cast<model::RawFunctionDefinition>(&Prototype))
    return usedRegisters(*Raw);
  else
    revng_abort("Layouts of non-function types are not supported.");
}

inline UsedRegisters
usedRegisters(const model::DefinitionReference &Prototype) {
  revng_assert(Prototype.isValid());
  return usedRegisters(*Prototype.getConst());
}

} // namespace abi::FunctionType

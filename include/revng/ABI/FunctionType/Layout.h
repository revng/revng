#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "llvm/ADT/STLExtras.h"

#include "revng/ABI/Definition.h"
#include "revng/ADT/STLExtras.h"
#include "revng/Model/Binary.h"
#include "revng/Support/YAMLTraits.h"

namespace abi::FunctionType {

/// Best effort `CABIFunctionDefinition` to `RawFunctionDefinition` conversion.
///
/// \note: this conversion is lossy since there's no way to represent some types
///        in `RawFunctionDefinition` in a reversible manner.
model::UpcastableType
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
    model::UpcastableType Type;
    llvm::SmallVector<model::Register::Values, 2> Registers;

    ReturnValue() = default;
    ReturnValue(model::UpcastableType &&Type) : Type(std::move(Type)) {}
  };

  struct Argument : public ReturnValue {
    using ReturnValue::ReturnValue;

  public:
    struct StackSpan {
      /// The offset should be interpreted as an offset within the struct
      /// containing the stack arguments. It's not an offset from some reference
      /// stack pointer value.
      uint64_t Offset = 0;
      uint64_t Size = 0;
    };

  public:
    std::optional<StackSpan> Stack;
    ArgumentKind::Values Kind = ArgumentKind::Scalar;
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
  static Layout make(const model::TypeDefinition &Prototype) {
    if (auto CABI = llvm::dyn_cast<model::CABIFunctionDefinition>(&Prototype))
      return Layout(*CABI);
    else if (auto *R = llvm::dyn_cast<model::RawFunctionDefinition>(&Prototype))
      return Layout(*R);
    else
      revng_abort("Layouts of non-function types are not supported.");
  }
  static Layout make(const model::UpcastableType &FunctionType) {
    revng_assert(!FunctionType.isEmpty());
    return make(FunctionType->toPrototype());
  }

public:
  bool verify() const debug_function;

  size_t argumentRegisterCount() const;
  size_t returnValueRegisterCount() const;
  llvm::SmallVector<model::Register::Values> argumentRegisters() const;
  llvm::SmallVector<model::Register::Values> returnValueRegisters() const;

  auto returnValueTypes() {
    return ReturnValues
           | std::views::transform([](ReturnValue &RV) { return RV.Type; });
  }

  auto returnValueTypes() const {
    return ReturnValues | std::views::transform([](const ReturnValue &RV) {
             return RV.Type;
           });
  }

  auto argumentTypes() {
    return Arguments
           | std::views::transform([](Argument &A) { return A.Type; });
  }

  auto argumentTypes() const {
    return Arguments
           | std::views::transform([](const Argument &A) { return A.Type; });
  }

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
    } else if (not ReturnValues[0].Type->isScalar()) {
      revng_assert(ReturnValues.size() == 1);
      return ReturnMethod::ModelAggregate;
    } else {
      revng_assert(ReturnValues.size() == 1);
      revng_assert(ReturnValues[0].Type->isScalar());
      return ReturnMethod::Scalar;
    }
  }

  // \note Call only if returnMethod() is ModelAggregate
  const model::Type &returnValueAggregateType() const {
    revng_assert(returnMethod() == ReturnMethod::ModelAggregate);

    if (hasSPTAR()) {
      revng_assert(ReturnValues.size() <= 1);
      revng_assert(Arguments.size() >= 1);
      return Arguments[0].Type->getPointee();
    } else {
      revng_assert(ReturnValues.size() == 1);
      revng_assert(!ReturnValues[0].Type->isScalar());
      return *ReturnValues[0].Type;
    }
  }

public:
  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Stream) const {
    // TODO: accept an arbitrary stream
    serialize(Stream, *this);
  }
};

inline Layout::Argument::StackSpan
operator+(const Layout::Argument::StackSpan &This, uint64_t Offset) {
  return { This.Offset + Offset, This.Size };
}

inline Layout::Argument::StackSpan
operator+(uint64_t Offset, const Layout::Argument::StackSpan &This) {
  return { This.Offset + Offset, This.Size };
}

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
calleeSavedRegisters(const model::UpcastableType &FunctionType) {
  revng_assert(!FunctionType.isEmpty());
  return calleeSavedRegisters(FunctionType->toPrototype());
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

inline uint64_t finalStackOffset(const model::UpcastableType &Prototype) {
  revng_assert(!Prototype.isEmpty());
  return finalStackOffset(Prototype->toPrototype());
}

/// A register holding (part of) an argument or return value, together with the
/// number of bytes the model says it actually carries.
struct UsedRegister {
  /// The register itself.
  model::Register::Values Location;

  /// The number of bytes the model assigns to this register. For a
  /// `RawFunctionDefinition` this is the size of the `Type` of the
  /// corresponding `NamedTypedRegister`, which can be smaller than the register
  /// width when only its low part carries data (e.g. a scalar living in a
  /// vector register). It dictates the width of the value `enforce-abi`
  /// materializes, so the C backend is never handed an oversized integer.
  uint64_t Size;

  UsedRegister(model::Register::Values Location, uint64_t Size) :
    Location(Location), Size(Size) {}

  /// Convenience conversion sizing a register by its full width. Used by the
  /// CABI path, which describes argument/return values by their position
  /// rather than by a per-register model `Type`.
  UsedRegister(model::Register::Values Location) :
    Location(Location), Size(model::Register::getSize(Location)) {}
};

struct UsedRegisters {
  llvm::SmallVector<UsedRegister> Arguments;
  llvm::SmallVector<UsedRegister> ReturnValues;
};
UsedRegisters usedRegisters(const model::CABIFunctionDefinition &Prototype);

inline UsedRegisters
usedRegisters(const model::RawFunctionDefinition &Prototype) {
  UsedRegisters Result;
  for (const model::NamedTypedRegister &Register : Prototype.Arguments())
    Result.Arguments.emplace_back(Register.Location(),
                                  *Register.Type()->size());
  for (const model::NamedTypedRegister &Register : Prototype.ReturnValues())
    Result.ReturnValues.emplace_back(Register.Location(),
                                     *Register.Type()->size());
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

inline UsedRegisters usedRegisters(const model::UpcastableType &FunctionType) {
  revng_assert(!FunctionType.isEmpty());
  return usedRegisters(FunctionType->toPrototype());
}

} // namespace abi::FunctionType

using FTL = abi::FunctionType::Layout;
namespace FTAK = abi::FunctionType::ArgumentKind;

template<>
struct llvm::yaml::ScalarEnumerationTraits<FTAK::Values>
  : public NamedEnumScalarTraits<FTAK::Values> {};

template<>
struct llvm::yaml::MappingTraits<FTL::Argument::StackSpan> {
  static void mapping(IO &IO, FTL::Argument::StackSpan &SS) {
    IO.mapRequired("Offset", SS.Offset);
    IO.mapRequired("Size", SS.Size);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(FTL::Argument::StackSpan)

template<>
struct llvm::yaml::MappingTraits<FTL::ReturnValue> {
  static void mapping(IO &IO, FTL::ReturnValue &RV) {
    IO.mapRequired("Type", RV.Type);
    IO.mapRequired("Registers", RV.Registers);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(FTL::ReturnValue)

template<>
struct llvm::yaml::MappingTraits<FTL::Argument> {
  static void mapping(IO &IO, FTL::Argument &A) {
    IO.mapRequired("Type", A.Type);
    IO.mapRequired("Kind", A.Kind);
    IO.mapRequired("Registers", A.Registers);
    IO.mapOptional("Stack", A.Stack);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(FTL::Argument)

template<>
struct llvm::yaml::MappingTraits<FTL> {
  static void mapping(IO &IO, FTL &L) {
    IO.mapRequired("Arguments", L.Arguments);
    IO.mapRequired("ReturnValues", L.ReturnValues);
    IO.mapRequired("CalleeSavedRegisters", L.CalleeSavedRegisters);
    IO.mapRequired("FinalStackOffset", L.FinalStackOffset);
  }
};

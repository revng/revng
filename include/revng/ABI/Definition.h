#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "revng/ABI/FunctionType/Support.h"
#include "revng/ABI/ScalarType.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Model/ABI.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/Register.h"
#include "revng/Support/Debug.h"
#include "revng/TupleTree/TupleTree.h"
#include "revng/TupleTree/TupleTreeDiff.h"

/* TUPLE-TREE-YAML

name: Definition
type: struct
doc: for the documentation see `references/abi-definition.md`
fields:
  - name: ABI
    type: model::ABI::Values

  - name: ArgumentsArePositionBased
    type: bool

  - name: OnlyStartDoubleArgumentsFromAnEvenRegister
    type: bool
  - name: ArgumentsCanBeSplitBetweenRegistersAndStack
    type: bool
  - name: BigArgumentsUsePointersToCopy
    type: bool
  - name: NoRegisterArgumentsCanComeAfterStackOnes
    type: bool
  - name: AllowUnnaturallyAlignedTypesInRegisters
    type: bool
  - name: CalleeIsResponsibleForStackCleanup
    type: bool
  - name: FloatsUseGPRs
    type: bool
  - name: StackArgumentsUseRegularStructAlignmentRules
    type: bool
  - name: UseStrictAggregateAlignmentRules
    type: bool

  - name: StackAlignment
    type: uint64_t
  - name: MinimumStackArgumentSize
    type: uint64_t
  - name: StackBytesAllocatedForRegisterArguments
    type: uint64_t
    optional: true

  - name: MaximumGPRsPerAggregateArgument
    type: uint64_t
  - name: MaximumGPRsPerAggregateReturnValue
    type: uint64_t
  - name: MaximumGPRsPerScalarArgument
    type: uint64_t
  - name: MaximumGPRsPerScalarReturnValue
    type: uint64_t

  - name: GeneralPurposeArgumentRegisters
    sequence:
      type: std::vector
      elementType: model::Register::Values
    optional: true
  - name: GeneralPurposeReturnValueRegisters
    sequence:
      type: std::vector
      elementType: model::Register::Values
    optional: true
  - name: VectorArgumentRegisters
    sequence:
      type: std::vector
      elementType: model::Register::Values
    optional: true
  - name: VectorReturnValueRegisters
    sequence:
      type: std::vector
      elementType: model::Register::Values
    optional: true
  - name: CalleeSavedRegisters
    sequence:
      type: std::vector
      elementType: model::Register::Values
    optional: true

  - name: ReturnValueLocationRegister
    type: model::Register::Values
    optional: true
  - name: ReturnValueLocationOnStack
    type: bool
    optional: true
  - name: ReturnValueLocationIsReturned
    type: bool
    optional: true

  - name: ScalarTypes
    sequence:
      type: SortedVector
      elementType: ScalarType

  - name: FloatingPointScalarTypes
    sequence:
      type: SortedVector
      elementType: ScalarType

key:
  - ABI

TUPLE-TREE-YAML */

#include "revng/ABI/Generated/Early/Definition.h"

namespace abi {

class Definition : public generated::Definition {
public:
  using generated::Definition::Definition;

public:
  static const Definition &get(model::ABI::Values ABI);

public:
  std::string_view getName() const { return model::ABI::getName(ABI()); }
  uint64_t getPointerSize() const { return model::ABI::getPointerSize(ABI()); }
  model::Architecture::Values getArchitecture() const {
    return model::ABI::getArchitecture(ABI());
  }

  /// Make sure current definition is valid.
  bool verify() const debug_function;

  /// Checks whether a given function type definition contradicts this ABI
  ///
  /// \note this is not an exhaustive check, so if it returns `false`,
  /// the function definitely is NOT compatible, but if it returns `true`
  /// it might either be compatible or not.
  ///
  /// \param RFT The function to check
  ///
  /// \return `false` if the function is definitely NOT compatible with the ABI,
  ///         `true` if it might be compatible.
  bool
  isPreliminarilyCompatibleWith(const model::RawFunctionDefinition &RFT) const;

  struct AlignmentInfo {
    uint64_t Value;
    bool IsNatural;
  };
  using AlignmentCache = std::unordered_map<const model::TypeDefinition *,
                                            AlignmentInfo>;

  /// Compute the natural alignment of the type in accordance with
  /// the current ABI
  ///
  /// \note  It mirrors, `model::TypeDefinition::size()` pretty closely, see
  /// documentation
  ///        related to it (and usage of the coroutines inside this codebase
  ///        in general) for more details on how it works.
  ///
  /// \param Type The type to compute the alignment of.
  /// \param ABI The ABI used to determine alignment of the primitive components
  ///        of the type
  ///
  /// \return either an alignment or a `std::nullopt` when it's not applicable.
  template<model::AnyType AnyType>
  std::optional<uint64_t> alignment(const AnyType &Type) const {
    AlignmentCache Cache;
    return alignment(Type, Cache);
  }

  template<model::AnyType AnyType>
  std::optional<bool> hasNaturalAlignment(const AnyType &Type) const {
    AlignmentCache Cache;
    return hasNaturalAlignment(Type, Cache);
  }

  std::optional<uint64_t> alignment(const model::Type &Type,
                                    AlignmentCache &Cache) const;
  std::optional<uint64_t> alignment(const model::TypeDefinition &Type,
                                    AlignmentCache &Cache) const;
  std::optional<bool> hasNaturalAlignment(const model::Type &Type,
                                          AlignmentCache &Cache) const;
  std::optional<bool>
  hasNaturalAlignment(const model::TypeDefinition &Definition,
                      AlignmentCache &Cache) const;

  uint64_t alignedOffset(uint64_t Offset, uint64_t Alignment) const {
    if (Offset == 0)
      return 0;

    revng_assert(llvm::isPowerOf2_64(Alignment));
    if (Offset % Alignment != 0)
      return Offset + Alignment - Offset % Alignment;

    return Offset;
  }

  template<model::AnyType AnyType>
  uint64_t alignedOffset(uint64_t Offset, const AnyType &Type) const {
    return alignedOffset(Offset, *alignment(Type));
  }

public:
  using RegisterSet = std::set<model::Register::Values>;

  /// Try to deduce the specific "holes" in the provided register state
  /// information.
  ///
  /// In short, when we have any information about arguments (for example, if
  /// we know that `r2` is used as a function argument) - we can extrapolate it
  /// to uncover more information about other register (in this example, that
  /// `r0` and `r1` must also either be active _unused_ arguments _or_ padding).
  /// This in information is embedded into the returned map.
  ///
  /// \returns `std::nullopt` if \ref State does not match the ABI (i.e. it
  ///          marks a non-argument register (like `r5` in the example used) as
  ///          an argument).
  std::optional<RegisterSet>
  tryDeducingArgumentRegisterState(RegisterSet &&Arguments) const;
  std::optional<RegisterSet>
  tryDeducingReturnValueRegisterState(RegisterSet &&ReturnValues) const;

  /// A more strict version of \ref tryDeducingArgumentRegisterState.
  ///
  /// The difference is that `tryDeducingArgumentRegisterState` expects all
  /// the input information to be 100% correct, with the most likely problem
  /// being the fact that we didn't detect ABI correctly (the original function
  /// uses one that differs from the one specified), while this one
  /// (`enforceArgumentRegisterState`) believes the ABI first and foremost,
  /// allowing this deduction to discard any contradicting data (for example
  /// if `r5` is specified as an argument, it's silently changed to `No`
  /// because ABI does not allow it to be).
  RegisterSet enforceArgumentRegisterState(RegisterSet &&Arguments) const;
  RegisterSet enforceReturnValueRegisterState(RegisterSet &&ReturnValues) const;

private:
  llvm::SmallVector<model::Register::Values, 8> argumentOrder() const {
    llvm::SmallVector<model::Register::Values, 8> Result;

    const auto &GPRs = GeneralPurposeArgumentRegisters();
    const model::Register::Values &RVL = ReturnValueLocationRegister();
    constexpr model::Register::Values Invalid = model::Register::Invalid;
    if (RVL != Invalid && !llvm::is_contained(GPRs, RVL))
      Result.emplace_back(RVL);

    for (auto Register : GPRs)
      if (!llvm::is_contained(Result, Register))
        Result.emplace_back(Register);
    for (auto Register : VectorArgumentRegisters())
      if (!llvm::is_contained(Result, Register))
        Result.emplace_back(Register);

    return Result;
  }

  llvm::SmallVector<model::Register::Values, 8> returnValueOrder() const {
    llvm::SmallVector<model::Register::Values, 8> Result;

    for (auto Register : GeneralPurposeReturnValueRegisters())
      if (!llvm::is_contained(Result, Register))
        Result.emplace_back(Register);
    for (auto Register : VectorReturnValueRegisters())
      if (!llvm::is_contained(Result, Register))
        Result.emplace_back(Register);

    return Result;
  }

  template<ranges::sized_range InputContainer,
           ranges::sized_range OutputContainer>
  void assertSortingWasSuccessful(std::string_view RegisterType,
                                  const InputContainer &Input,
                                  const OutputContainer &Output) const {
    if (Input.size() != Output.size()) {
      std::string Error = "Unable to sort " + std::string(RegisterType)
                          + " registers.\nMost likely some of the present "
                            "registers are not allowed to be used under "
                            "the current ABI ("
                          + std::string(getName())
                          + ").\nList of registers to be sorted: [ ";
      if (Input.size() != 0) {
        for (auto Register : Input)
          Error += model::Register::getName(Register).str() + ", ";
        Error.resize(Error.size() - 2);
      }

      Error += " ]\nSorted list: [ ";
      if (Output.size() != 0) {
        for (auto Register : Output)
          Error += model::Register::getName(Register).str() + ", ";
        Error.resize(Error.size() - 2);
      }
      Error += " ]\n";
      revng_abort(Error.c_str());
    }
  }

public:
  template<ranges::sized_range Container>
  llvm::SmallVector<model::Register::Values, 8>
  sortArguments(const Container &Registers) const {
    SortedVector<model::Register::Values> Lookup;
    {
      auto Inserter = Lookup.batch_insert();
      for (auto &&Register : Registers)
        Inserter.insert(Register);
    }

    llvm::SmallVector<model::Register::Values, 8> Result;
    for (auto Register : argumentOrder())
      if (Lookup.contains(Register))
        Result.emplace_back(Register);

    assertSortingWasSuccessful("argument", Registers, Result);
    return Result;
  }

  template<ranges::sized_range Container>
  llvm::SmallVector<model::Register::Values, 8>
  sortReturnValues(const Container &Registers) const {
    SortedVector<model::Register::Values> Lookup;
    {
      auto Inserter = Lookup.batch_insert();
      for (auto &&Register : Registers)
        Inserter.insert(Register);
    }

    llvm::SmallVector<model::Register::Values, 8> Result;
    for (auto Register : returnValueOrder())
      if (Lookup.contains(Register))
        Result.emplace_back(Register);

    assertSortingWasSuccessful("return value", Registers, Result);
    return Result;
  }

  /// Takes care of extending (padding) the size of a stack argument.
  ///
  /// \note This only accounts for the post-padding (extension).
  ///       Pre-padding (offset) needs to be taken care of separately.
  ///
  /// \param Size The size of the argument without the padding.
  ///
  /// \return The size of the argument with the padding.
  uint64_t paddedSizeOnStack(uint64_t Size) const {
    return FunctionType::paddedSizeOnStack(Size, MinimumStackArgumentSize());
  }
};

} // namespace abi

#include "revng/ABI/Generated/Late/Definition.h"

#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "revng/ABI/Definition.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Register.h"

namespace abi::FunctionType {

using RegisterVector = llvm::SmallVector<model::Register::Values, 2>;

/// The internal representation of the argument shared between both
/// the to-raw conversion and the layout.
struct DistributedValue {
  /// A list of registers the argument uses.
  RegisterVector Registers = {};

  /// The total size of the argument (including padding if necessary) in bytes.
  uint64_t Size = 0;

  /// The size of the piece of the argument placed on the stack.
  /// \note: has to be equal to `0` or `this->Size` for any ABI for which
  ///        `abi::Definition::ArgumentsCanBeSplitBetweenRegistersAndStack()`
  ///        returns `false`. Otherwise, it has to be an integer value, such
  ///        that `(0 <= SizeOnStack <= this->Size)` is true.
  uint64_t SizeOnStack = 0;

  /// Mark this argument as a padding argument, which means an unused location
  /// (either a register or a piece of the stack) which needs to be seen as
  /// a separate argument to be able to place all the following arguments
  /// in the correct positions.
  ///
  /// The "padding" arguments are emitted as normal arguments in RawFunctionType
  /// but are omitted in `Layout`.
  bool RepresentsPadding = false;
};
using DistributedValues = llvm::SmallVector<DistributedValue, 8>;

using RegisterSpan = std::span<const model::Register::Values>;

class ValueDistributor {
public:
  const abi::Definition &ABI;
  uint64_t UsedGeneralPurposeRegisterCount = 0;
  uint64_t UsedVectorRegisterCount = 0;
  uint64_t UsedStackOffset = 0;
  uint64_t ArgumentIndex = 0;

protected:
  explicit ValueDistributor(const abi::Definition &ABI) : ABI(ABI) {
    revng_assert(ABI.verify());
  }

  /// Helper for converting a single object into a "distributed" state.
  ///
  /// \param Type The type of the object.
  /// \param Registers The list of registers allowed for usage for the type.
  /// \param OccupiedRegisterCount The count of registers in \ref Registers
  ///        Container that are already occupied.
  /// \param AllowedRegisterLimit The maximum number of registers available to
  ///        use for the current argument.
  /// \param ForbidSplittingBetweenRegistersAndStack Allows overriding the
  ///        \ref abi::Definition::ArgumentsCanBeSplitBetweenRegistersAndStack
  ///        of the ABI for the sake of current distribution. This should be set
  ///        to `true` when return value is being distributed.
  ///
  /// \returns A pair consisting of at least one `DistributedValue` object (if
  ///          multiple objects are returned, only one of them corresponds to
  ///          the specified argument: all the other ones represent padding) and
  ///          the new count of occupied registers, after the current argument.
  std::pair<DistributedValues, uint64_t>
  distribute(model::QualifiedType Type,
             RegisterSpan Registers,
             uint64_t OccupiedRegisterCount,
             uint64_t AllowedRegisterLimit,
             bool ForbidSplittingBetweenRegistersAndStack);
};

class ArgumentDistributor : public ValueDistributor {
public:
  explicit ArgumentDistributor(const abi::Definition &ABI) :
    ValueDistributor(ABI){};

  DistributedValues nextArgument(const model::QualifiedType &ArgumentType);

private:
  DistributedValues positionBased(const model::QualifiedType &Type);
  DistributedValues nonPositionBased(const model::QualifiedType &Type);
};

class ReturnValueDistributor : public ValueDistributor {
public:
  explicit ReturnValueDistributor(const abi::Definition &ABI) :
    ValueDistributor(ABI){};

  DistributedValue returnValue(const model::QualifiedType &ReturnValueType);
};

} // namespace abi::FunctionType

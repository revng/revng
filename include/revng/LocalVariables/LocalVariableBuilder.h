#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/Model/BinaryIdentifier.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/OpaqueFunctionsPool.h"

namespace llvm {

class AllocaInst;
class CallInst;
class Function;
class Instruction;
class IntegerType;
class LoadInst;
class Module;
class StoreInst;
class Type;
class Value;

} // end namespace llvm

namespace model {

class Binary;
class Type;

} // end namespace model

/// Types used by the LocalVariableBuilder for its internal operations.
struct VariableBuilderTypes {
  /// An LLVM integer type whose size matches the size of a pointer in the
  /// Binary we're decompiling.
  llvm::IntegerType *InputPointerSizedInteger = nullptr;

  /// An LLVM integer type whose size matches the size of a pointer in the
  /// target architecture
  llvm::IntegerType *TargetPointerSizedInteger = nullptr;

  /// An LLVM 8-bits integer
  llvm::IntegerType *Int8Ty = nullptr;

public:
  /// Constructor from Model and an LLVM Module.
  /// This is the constructor that is used in all cases except for unit testing,
  /// where we want to decouple from the model.
  VariableBuilderTypes(const model::Binary &TheBinary, llvm::Module &TheModule);

  /// Constructor from LLVM Module, with explicit InputPointerByteSize.
  /// This is meant to be used only for unit testing, in situations where we
  /// want to decouple the tests from the Model.
  VariableBuilderTypes(llvm::Module &TheModule, unsigned InputPointerByteSize);
};

/// Various LLVM Passes in the decompilation pipelines need to create local
/// variables and read/write memory accesses from/to them. In the legacy
/// decompilation pipeline these were represented by dedicated functions, tagged
/// with specific FunctionTags, to represent dedicated opcodes without using
/// LLVM intrinsics. This workaround with FunctionTags and custom opcodes is
/// scheduled to be dropped for the clift-based decompilation pipeline based on
/// the Clift MLIR dialect, which will use regular LLVM alloca/load/store
/// instructions.
///
/// This class provides a bunch of helpers to deal with creation of local
/// variables. The IsLegacy field is used to select at compile-time the
/// appropriate mode of operation:
/// - IsLegacy == true: uses the old FunctionTags and dedicated functions to
///   represent dedicated opcodes
/// - IsLegacy == false: uses regular LLVM alloca/load/store instructions
//
// TODO: when the migration is over, the IsLegacy field can be dropped to
// fully embrace the new ways.
template<bool IsLegacy>
class LocalVariableBuilder {
public:
  using AllocaInst = llvm::AllocaInst;
  using CallInst = llvm::CallInst;
  using LoadInst = llvm::LoadInst;
  using StoreInst = llvm::StoreInst;
  using Value = llvm::Value;

  using AssignType = std::conditional_t<IsLegacy, CallInst, StoreInst>;
  using CopyType = std::conditional_t<IsLegacy, CallInst, LoadInst>;
  using LocalVarType = std::conditional_t<IsLegacy, CallInst, AllocaInst>;
  using ReferenceType = std::conditional_t<IsLegacy, CallInst, Value>;

private:
  /// The types necessary for this LocalVariableBuilder to operate.
  VariableBuilderTypes Types;

  /// The module that this class manipulates.
  /// llvm::Module &M;

  /// The function where this helper inserts local variables.
  llvm::Function *F = nullptr;

private:
  /// Constructor that also initializes the target function \a F. This is
  /// private so it can only be called by the associated public factory.
  LocalVariableBuilder(VariableBuilderTypes TheTypes, llvm::Function *F) :
    Types(TheTypes), F(F) {}

  /// Constructor that leaves the target function unset. This is private so it
  /// can only be called by the associated public factory.
  LocalVariableBuilder(VariableBuilderTypes TheTypes) :
    LocalVariableBuilder(TheTypes, nullptr) {}

public:
  /// Factory method for non-legacy mode, which also sets the target function to
  /// \a F.
  static LocalVariableBuilder
  make(VariableBuilderTypes TheTypes, llvm::Function *F)
    requires(not IsLegacy)
  {
    return LocalVariableBuilder(TheTypes, F);
  }

  /// Factory method for non-legacy mode.
  static LocalVariableBuilder make(VariableBuilderTypes TheTypes)
    requires(not IsLegacy)
  {
    return make(TheTypes, nullptr);
  }

public:
  ~LocalVariableBuilder() = default;

  LocalVariableBuilder(const LocalVariableBuilder &) = default;

  LocalVariableBuilder(LocalVariableBuilder &&) = default;

public:
  /// Sets the function where the LocalVariableBuilder injects instructions
  /// representing local variables.
  void setTargetFunction(llvm::Function *NewF) { F = NewF; }

  /// Creates an llvm::Instruction that models the allocation of a local
  /// variable.
  /// The created instruction is inserted at the beginning of the function F.
  /// This is typically an alloca, but it's a call to LocalVariable in legacy
  /// mode.
  //
  // TODO: this method can become const when we drop legacy mode because we'll
  // not be using OpaqueFunctionsPool anymore.
  LocalVarType *createLocalVariable(const model::Type &VariableType);

  /// Methods meant to be used only by SegregateStackAccesses.
  /// TODO: eventually, when we drop legacy mode, the whole LocalVariableBuilder
  /// will be only used by SegregateStackAccesses.
  ///
  ///@{

  /// Creates an llvm::Instruction that models the allocation of a local
  /// variable, and takes its address.
  /// The created instruction is inserted at the beginning of the function F.
  /// This is typically an alloca, but it's a call to LocalVariable in legacy
  /// mode.
  ///
  /// In legacy mode:
  /// - the instruction to allocate the local variable is a custom opaque
  ///   function
  /// - it's address is taken with AddressOf, whose type on LLVM is a
  ///   pointer-sized integer type
  ///
  /// In non-legacy mode:
  /// - the instruction to allocate the local variable is a regular alloca
  /// - the alloca is ptr-to-int casted to a pointer-sized integer.custom opaque
  //
  // TODO: this method can become const when we drop legacy mode because we'll
  // not be using OpaqueFunctionsPool anymore.
  std::pair<LocalVarType *, llvm::Instruction *>
  createLocalVariableAndTakeIntAddress(const model::Type &VariableType);

  /// Creates an alloca in \a F with type \a T.
  /// Allocas created with this method are intended to be inserted temporarily,
  /// and subsequently optimized away from LLVM optimizations.
  /// There's no need to tag them with model::Types in any way.
  std::pair<llvm::AllocaInst *, llvm::Value *>
  createAllocaWithPtrToInt(llvm::Function *F, llvm::Type *T) const;

  /// Creates an llvm::Instruction that models the allocation of a local
  /// variable representing the stack frame, and takes its address.
  /// The returned llvm::Instruction has an integer type on LLVM, and its size
  /// is equal to the size of a pointer in the associated Model.Architecture.
  /// The instruction that represents the allocation of the local variable is
  /// inserted at the beginning of function F, after all the allocas.
  ///
  /// In legacy mode:
  /// - the instruction to allocate the local variable is a custom opaque
  ///   function
  /// - it's address is taken with AddressOf, whose type on LLVM is a
  ///   pointer-sized integer type
  ///
  /// In non-legacy mode:
  /// - the instruction to allocate the local variable is a regular alloca
  /// - the alloca is ptr-to-int casted to a pointer-sized integer.custom opaque
  //
  // TODO: can this method become const when we drop legacy mode?
  llvm::Instruction *createStackFrameVariable(model::UpcastableType FrameType);

  /// Creates an llvm::Instruction that models the allocation of a local
  /// variable to be passed as stack argument to a call instruction, and take
  /// its address.
  /// The returned llvm::Instruction has an integer type on LLVM, and its size
  /// is equal to the size of a pointer in the associated Model.Architecture.
  /// The instruction that represents the allocation of the local variable is
  /// inserted at the beginning of function F, after all the allocas.
  ///
  /// In legacy mode:
  /// - the instruction to allocate the local variable is a custom opaque
  ///   function
  /// - it's address is taken with AddressOf, whose type on LLVM is a
  ///   pointer-sized integer type
  ///
  /// In non-legacy mode:
  /// - the instruction to allocate the local variable is a regular alloca
  /// - the alloca is ptr-to-int casted to a pointer-sized integer.custom opaque
  //
  // TODO: this method can be dropped when we drop legacy mode, because the
  // callers can just switch to call createLocalVariableAndTakeIntAddress
  llvm::Instruction *
  createCallStackArgumentVariable(const model::Type &VariableType);

  ///@}
};

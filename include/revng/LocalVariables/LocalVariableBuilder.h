#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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
///
/// TODO: when the migration is over, the IsLegacy field can be dropped to
/// fully embrace the new ways.
template<bool IsLegacy>
class LocalVariableBuilder {
public:
  using TypePair = FunctionTags::TypePair;

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
  const model::Binary &Binary;

  /// The module that this class manipulates.
  llvm::Module &M;

  /// An LLVM integer type whose size matches the size of a pointer in the
  /// Binary we're decompiling.
  llvm::IntegerType *InputPointerSizedInteger = nullptr;

  /// An LLVM integer type whose size matches the size of a pointer in the
  /// target architecture
  llvm::IntegerType *TargetPointerSizedInteger = nullptr;

  /// An LLVM 8-bits integer
  llvm::IntegerType *Int8Ty = nullptr;

  /// The function where this helper inserts local variables.
  llvm::Function *F = nullptr;

  /// \name Data members used only for Legacy mode.
  ///
  /// TODO: drop these when we drop legacy mode
  ///
  ///@{

  /// Pool of functions that represent allocation of local variables
  OpaqueFunctionsPool<llvm::Type *> LocalVarPool;

  /// Pool of functions that represent the operation that assigns a value to a
  /// local variables
  OpaqueFunctionsPool<llvm::Type *> AssignPool;

  /// Pool of functions that represent the operation that copies a value from a
  /// local variables and returns a temporary
  OpaqueFunctionsPool<llvm::Type *> CopyPool;

  /// Pool of functions that represent the AddressOf operator.
  /// This is a pointer to a pool, because users of of LocalVariableBuilders
  /// typically want to inject other AddressOf operations, which means that they
  /// need to have a shared pool of AddressOf, in order not to go out of sync.
  OpaqueFunctionsPool<TypePair> &AddressOfPool;

  /// LLVM Function used to represent the allocation of the stack frame.
  llvm::Function *StackFrameAllocator = nullptr;

  /// LLVM Function used to represent the allocation of the stack arguments for
  /// a call to an isolated function.
  llvm::Function *CallStackArgumentsAllocator = nullptr;

  ///@}

private:
  LocalVariableBuilder(const model::Binary &TheBinary,
                       llvm::Module &TheModule,
                       OpaqueFunctionsPool<TypePair> *TheAddressOfPool);

  LocalVariableBuilder(const model::Binary &TheBinary,
                       llvm::Module &TheModule) :
    LocalVariableBuilder(TheBinary, TheModule, nullptr) {}

public:
  ~LocalVariableBuilder() = default;

  LocalVariableBuilder(const LocalVariableBuilder &) = default;

  LocalVariableBuilder(LocalVariableBuilder &&) = default;

  bool isLegacy() const { return IsLegacy; }

public:
  /// Factories to create LocalVariableBuilders
  ///
  /// TODO: drop these when we drop legacy mode, in favor of just using the only
  /// constructor left.
  ///
  ///@{

  static LocalVariableBuilder
  makeLegacyStackBuilder(const model::Binary &TheBinary,
                         llvm::Module &TheModule,
                         OpaqueFunctionsPool<TypePair> &TheAddressOfPool)
    requires IsLegacy
  {
    return LocalVariableBuilder<IsLegacy>(TheBinary,
                                          TheModule,
                                          &TheAddressOfPool);
  }

  static LocalVariableBuilder make(const model::Binary &TheBinary,
                                   llvm::Module &TheModule) {
    return LocalVariableBuilder<IsLegacy>(TheBinary, TheModule);
  }

  static LocalVariableBuilder make(const model::Binary &TheBinary,
                                   llvm::Function &TheFunction) {
    LocalVariableBuilder<IsLegacy> LVB(TheBinary, *TheFunction.getParent());
    LVB.setTargetFunction(&TheFunction);
    return LVB;
  }

  ///@}

public:
  /// Sets the function where the LocalVariableBuilder injects instructions
  /// representing local variables.
  void setTargetFunction(llvm::Function *NewF) { F = NewF; }

  /// Creates an llvm::Instruction that models the allocation of a local
  /// variable.
  /// The created instruction is inserted at the beginning of the function F.
  /// This is typically an alloca, but it's a call to LocalVariable in legacy
  /// mode.
  /// TODO: this method can become const when we drop legacy mode.
  LocalVarType *createLocalVariable(const model::Type &VariableType);

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
  ///
  /// TODO: this method can become const when we drop legacy mode.
  std::pair<LocalVarType *, llvm::Instruction *>
  createLocalVariableAndTakeIntAddress(const model::Type &VariableType);

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
  ///
  /// TODO: this method can become const when we drop legacy mode.
  llvm::Instruction *createStackFrameVariable();

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
  ///
  /// TODO: this method can be dropped when we drop legacy mode, because the
  /// callers can just switch to call createLocalVariableAndTakeAddress
  llvm::Instruction *
  createCallStackArgumentVariable(const model::Type &VariableType);

public:
  /// Takes an instruction representing a variable location and a Use, and
  /// replaces the Use with a copy instruction from the instruction representing
  /// the variable location
  ///
  /// In legacy mode an instruction representing a variable location should be
  /// a call to an opaque function tagged with FunctionTags::IsRef. A copy
  /// instruction is a call to Copy.
  /// In non-legacy mode an instruction representing a variable location should
  /// be a ptr-typed instruction, and copy is a LoadInst.
  ///
  /// TODO: This method can be made const whenever we drop legacy mode.
  CopyType *createCopyOnUse(ReferenceType *LocationToCopy, llvm::Use &U);

  /// Takes an assignment instruction and a Use and replaces the Use with a
  /// newly created copy of the location assigned by the assignment instruction.
  ///
  /// In legacy mode an assignment instruction is a call to Assign and a copy
  /// instruction is a call to Copy.
  /// In non-legacy mode an assignment instruction is just a StoreInst, and copy
  /// a LoadInst.
  ///
  /// TODO: This method can be made const whenever we drop legacy mode.
  CopyType *createCopyFromAssignedOnUse(AssignType *Assign, llvm::Use &U);

  /// Creates an assignment instruction, at the location specified by
  /// InsertBefore, assigning ValueToAssign to the location represented by
  /// LocationToAssign.
  ///
  /// In legacy mode an assignment instruction is a call to Assign.
  /// In non-legacy mode an assignment instruction is just a StoreInst.
  ///
  /// TODO: This method can be made const whenever we drop legacy mode.
  AssignType *createAssignmentBefore(llvm::Value *LocationToAssign,
                                     llvm::Value *ValueToAssign,
                                     llvm::Instruction *InsertBefore);

  /// Creates an alloca in \a F with type \a T.
  /// Allocas created with this method are intended to be inserted temporarily,
  /// and subsequently optimized away from LLVM optimizations.
  /// There's no need to tag them with model::Types in any way.
  std::pair<llvm::AllocaInst *, llvm::Value *>
  createAllocaWithPtrToInt(llvm::Function *F, llvm::Type *T) const;

private:
  /// Takes an assignment instruction and returns its operand that represents
  /// the assigned location.
  /// In legacy mode an assignment instruction is a call to Assign.
  /// In non-legacy mode an assignment instruction is just a StoreInst.
  ReferenceType *getAssignedLocation(AssignType *Assign) const;

  /// Legacy methods for lazily initializing the StackFrameAllocator and
  /// CallStackArgumentsAllocator, in Legacy mode.
  ///
  /// TODO: drop these when we drop legacy mode
  ///@{

  llvm::Function *getStackFrameAllocator();

  llvm::Function *getCallStackArgumentsAllocator();

  ///@}
};

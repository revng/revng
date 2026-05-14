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

/// OpaqueFunctionPools used by the LocalVariableBuilder in legacy mode.
//
// TODO: drop this when we drop legacy mode.
struct LegacyCustomFunctions {
  /// The LLVM module holding all the functions belonging to the various pools.
  llvm::Module &M;

  /// Pool of functions that represent allocation of local variables
  OpaqueFunctionsPool<llvm::Type *> LocalVarPool;

  /// Pool of functions that represent the operation that assigns a value to a
  /// local variables
  OpaqueFunctionsPool<llvm::Type *> AssignPool;

  /// Pool of functions that represent the operation that copies a value from a
  /// local variables and returns a temporary
  OpaqueFunctionsPool<llvm::Type *> CopyPool;

  /// Pool of functions that represent the AddressOf operator.
  OpaqueFunctionsPool<FunctionTags::TypePair> AddressOfPool;

private:
  /// Initialize from an llvm::Module.
  /// This is meant to be used only in legacy mode, so that the non-legacy mode
  /// is entirely decoupled from the custom functions. This is why it's private,
  /// so it's only accessible via the `makeLegacy` factory method.
  LegacyCustomFunctions(llvm::Module &TheModule);

public:
  /// Factory meant to be used only in legacy mode
  static LegacyCustomFunctions makeLegacy(llvm::Module &TheModule) {
    return LegacyCustomFunctions(TheModule);
  }
};

/// Custom opaque functions used to represent stack allocations in legacy mode.
//
// TODO: drop this when we drop legacy mode.
struct LegacyStackAllocators {

  /// LLVM Module holding all the allocator functions.
  llvm::Module &M;

  /// LLVM Function used to represent the allocation of the stack frame.
  llvm::Function *StackFrameAllocator = nullptr;

  /// LLVM Function used to represent the allocation of the stack arguments for
  /// a call to an isolated function.
  llvm::Function *CallStackArgumentsAllocator = nullptr;

public:
  // Delete the default constructor, to force construction via the factory
  LegacyStackAllocators() = delete;

private:
  /// Initialize from an llvm::Module.
  /// This is meant to be used only in legacy mode, so that the non-legacy mode
  /// is entirely decoupled from the model and from the custom functions. This
  /// is why it's private, so it's only accessible via the `makeLegacy` factory
  /// method.
  LegacyStackAllocators(llvm::IntegerType *InputPointerSizedInteger,
                        llvm::Module &TheModule);

public:
  /// Factory meant to be used only in legacy mode.
  static LegacyStackAllocators makeLegacy(const model::Binary &TheBinary,
                                          llvm::Module &TheModule);
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

  /// Pointers to the custom function pools necessary in legacy mode.
  ///
  // TODO: drop when we drop legacy mode
  std::optional<LegacyCustomFunctions> CustomFunctions;

  /// Pointers to the custom allocator functions necessary in legacy mode.
  //
  // TODO: drop when we drop legacy mode.
  std::optional<LegacyStackAllocators> Allocators;

private:
  /// Constructor for non-legacy mode, that leaves the custom functions pools
  /// and the allocators not initialized, but initializes the target function \a
  /// F. This is private so it can only be called by the associated public
  /// factory, which is only available when IsLegacy is false.
  LocalVariableBuilder(VariableBuilderTypes TheTypes, llvm::Function *F) :
    Types(TheTypes),
    F(F),
    CustomFunctions(std::nullopt),
    Allocators(std::nullopt) {}

  /// Constructor for non-legacy mode, that leaves the custom functions pools
  /// and the allocators not initialized This is private so it can only be
  /// called by the associated public factory, which is only available when
  /// IsLegacy is false.
  LocalVariableBuilder(VariableBuilderTypes TheTypes) :
    LocalVariableBuilder(TheTypes, nullptr) {}

public:
  /// Factory method for non-legacy mode, which also sets the target function to
  /// \a F. This is private so it can only be called by the associated public
  /// factory, which is only available when IsLegacy is false.
  static LocalVariableBuilder
  make(VariableBuilderTypes TheTypes, llvm::Function *F)
    requires(not IsLegacy)
  {
    return LocalVariableBuilder(TheTypes, F);
  }

  /// Factory method for non-legacy mode.
  /// This is private so it can only be called by the associated public factory,
  /// which is only available when IsLegacy is false.
  static LocalVariableBuilder make(VariableBuilderTypes TheTypes)
    requires(not IsLegacy)
  {
    return make(TheTypes, nullptr);
  }

private:
  /// Constructor for legacy mode, that initializes the custom functions pools
  /// and the allocators from the Model and an llvm::Module.
  LocalVariableBuilder(const model::Binary &TheBinary,
                       llvm::Module &TheModule) :
    Types(VariableBuilderTypes(TheBinary, TheModule)),
    F(nullptr),
    CustomFunctions(LegacyCustomFunctions::makeLegacy(TheModule)),
    Allocators(LegacyStackAllocators::makeLegacy(TheBinary, TheModule)) {}

  /// Constructor for legacy mode, that initializes the custom functions pools
  /// and the allocators from the Model and an llvm::Function, while also
  /// setting to the target function \a F.
  LocalVariableBuilder(const model::Binary &TheBinary, llvm::Function *TheF) :
    LocalVariableBuilder(TheBinary, *TheF->getParent()) {
    F = TheF;
  }

public:
  /// Factory method for legacy mode.
  /// This is private so it can only be called by the associated public factory,
  /// which is only available when IsLegacy is true.
  //
  // TODO: we can drop this when we drop legacy mode.
  static LocalVariableBuilder
  makeLegacy(const model::Binary &TheBinary, llvm::Module &TheModule)
    requires IsLegacy
  {
    return LocalVariableBuilder(TheBinary, TheModule);
  }

  /// Factory method for legacy mode, which also sets the target function to \a
  /// F. This is private so it can only be called by the associated public
  /// factory, which is only available when IsLegacy is true.
  //
  // TODO: we can drop this when we drop legacy mode.
  static LocalVariableBuilder
  makeLegacy(const model::Binary &TheBinary, llvm::Function *F)
    requires IsLegacy
  {
    return LocalVariableBuilder(TheBinary, F);
  }

public:
  ~LocalVariableBuilder() = default;

  LocalVariableBuilder(const LocalVariableBuilder &) = default;

  LocalVariableBuilder(LocalVariableBuilder &&) = default;

public:
  /// Sets the function where the LocalVariableBuilder injects instructions
  /// representing local variables.
  void setTargetFunction(llvm::Function *NewF) {
    if constexpr (IsLegacy) {
      revng_assert(CustomFunctions.has_value());
      revng_assert(Allocators.has_value());
      revng_assert(&Allocators->M == &CustomFunctions->M);
      revng_assert(&Allocators->M == NewF->getParent());
    } else {
      revng_assert(not CustomFunctions.has_value());
      revng_assert(not Allocators.has_value());
    }
    F = NewF;
  }

  /// Returns a reference to the AddressOf pool, in case the owner of the
  /// LocalVariableBuilder needs to add other calls to AddressOf.
  //
  // TODO: drop this when we drop legacy mode.
  OpaqueFunctionsPool<FunctionTags::TypePair> *getAddressOfPool() {
    if constexpr (IsLegacy)
      return &CustomFunctions.value().AddressOfPool;

    revng_assert(not CustomFunctions.has_value());
    return nullptr;
  }

  /// Creates an llvm::Instruction that models the allocation of a local
  /// variable.
  /// The created instruction is inserted at the beginning of the function F.
  /// This is typically an alloca, but it's a call to LocalVariable in legacy
  /// mode.
  //
  // TODO: this method can become const when we drop legacy mode because we'll
  // not be using OpaqueFunctionsPool anymore.
  LocalVarType *createLocalVariable(const model::Type &VariableType);

  /// Takes an instruction representing a variable location and a Use, and
  /// replaces the Use with a copy instruction from the instruction representing
  /// the variable location
  ///
  /// In legacy mode an instruction representing a variable location should be
  /// a call to an opaque function tagged with FunctionTags::IsRef. A copy
  /// instruction is a call to Copy.
  /// In non-legacy mode an instruction representing a variable location should
  /// be a ptr-typed instruction, and copy is a LoadInst.
  //
  // TODO: this method can become const when we drop legacy mode because we'll
  // not be using OpaqueFunctionsPool anymore.
  CopyType *createCopyOnUse(ReferenceType *LocationToCopy, llvm::Use &U);

  /// Takes an assignment instruction and a Use and replaces the Use with a
  /// newly created copy of the location assigned by the assignment instruction.
  ///
  /// In legacy mode an assignment instruction is a call to Assign and a copy
  /// instruction is a call to Copy.
  /// In non-legacy mode an assignment instruction is just a StoreInst, and copy
  /// a LoadInst.
  //
  // TODO: this method can become const when we drop legacy mode because we'll
  // not be using OpaqueFunctionsPool anymore.
  CopyType *createCopyFromAssignedOnUse(AssignType *Assign, llvm::Use &U);

  /// Creates an assignment instruction, at the location specified by
  /// InsertBefore, assigning ValueToAssign to the location represented by
  /// LocationToAssign.
  ///
  /// In legacy mode an assignment instruction is a call to Assign.
  /// In non-legacy mode an assignment instruction is just a StoreInst.
  //
  // TODO: this method can become const when we drop legacy mode because we'll
  // not be using OpaqueFunctionsPool anymore.
  AssignType *createAssignmentBefore(llvm::Value *LocationToAssign,
                                     llvm::Value *ValueToAssign,
                                     llvm::Instruction *InsertBefore);

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

private:
  /// Takes an assignment instruction and returns its operand that represents
  /// the assigned location.
  /// In legacy mode an assignment instruction is a call to Assign.
  /// In non-legacy mode an assignment instruction is just a StoreInst.
  ReferenceType *getAssignedLocation(AssignType *Assign) const;

  /// Legacy methods for lazily initializing the StackFrameAllocator and
  /// CallStackArgumentsAllocator, in Legacy mode.
  ///
  ///@{

  // TODO: drop these when we drop legacy mode

  llvm::Function *getStackFrameAllocator();

  llvm::Function *getCallStackArgumentsAllocator();
};

#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"

#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/FunctionTags.h"

/// Returns true if I should be considered with side effects for decompilation
/// purposes.
inline bool hasSideEffects(const llvm::Instruction &I) {

  // StoreInst has side effects
  if (llvm::isa<llvm::StoreInst>(I))
    return true;

  auto *Call = llvm::dyn_cast<llvm::CallInst>(&I);

  // Besides StoreInst, only calls may have side effects.
  if (not Call)
    return false;

  // All call to isolated functions have side effects.
  if (isCallToIsolatedFunction(Call))
    return true;

  // All calls for which we cannot establish the callee have side effects.
  auto *Callee = Call->getCalledFunction();
  if (not Callee)
    return true;

  // If the callee is an intrinsic use MemoryEffects to decide if it has side
  // effects.
  if (Callee->isIntrinsic()) {
    bool IsReadOnly = Callee->onlyReadsMemory();
    bool IsReadNone = Callee->doesNotAccessMemory();
    return not IsReadOnly and not IsReadNone;
  }

  // QEMU stuff and exceptional functions have side effects.
  if (FunctionTags::Helper.isTagOf(Callee) or FunctionTags::QEMU.isTagOf(Callee)
      or FunctionTags::Exceptional.isTagOf(Callee))
    return true;

  // Stuff that writes to memory has side effects
  if (FunctionTags::Assign.isTagOf(Callee)
      or FunctionTags::WritesMemory.isTagOf(Callee))
    return true;

  // Functions representing custom opcodes with reference semantics never have
  // side effects, since under the syntactic sugar or reference they only encode
  // pointer arithmetic.
  if (FunctionTags::IsRef.isTagOf(Callee)
      or FunctionTags::AllocatesLocalVariable.isTagOf(Callee))
    return false;

  // AddressOf never has side effects, since it's just taking the address.
  if (FunctionTags::AddressOf.isTagOf(Callee))
    return false;

  // Literal print decorators never have side effects, since they are
  // just metadata on how to print constants.
  if (FunctionTags::LiteralPrintDecorator.isTagOf(Callee)
      or FunctionTags::StringLiteral.isTagOf(Callee))
    return false;

  // Functions that represent custom opcodes with arithmetic or bitwise
  // semantics never have side effects.
  if (FunctionTags::UnaryMinus.isTagOf(Callee)
      or FunctionTags::BinaryNot.isTagOf(Callee)
      or FunctionTags::BooleanNot.isTagOf(Callee))
    return false;

  // Casts and parentheses never have side effects
  if (FunctionTags::ModelCast.isTagOf(Callee)
      or FunctionTags::Parentheses.isTagOf(Callee))
    return false;

  // Opaque extract values and struct initializers never have side effects
  if (FunctionTags::OpaqueExtractValue.isTagOf(Callee)
      or FunctionTags::StructInitializer.isTagOf(Callee))
    return false;

  // Opaque register values never have side effects
  if (FunctionTags::OpaqueCSVValue.isTagOf(Callee))
    return false;

  // Stuff that reads from memory never has side effects
  if (FunctionTags::Copy.isTagOf(Callee)
      or FunctionTags::ReadsMemory.isTagOf(Callee))
    return false;

  return true;
}

/// Check if \a ModelType can be assigned to an llvm::Value of type \a LLVMType
/// during a memory operations (load, store and the like).
inline bool areMemOpCompatible(const model::QualifiedType &ModelType,
                               const llvm::Type &LLVMType,
                               const model::Binary &Model) {

  // We don't load or store entire structs in a single mem operation
  if (not ModelType.isScalar())
    return false;

  // loads/stores from/to void pointers are not allowed
  if (ModelType.isVoid())
    return false;

  auto ModelSize = ModelType.size().value();

  // For LLVM pointers, we want to check that the model type has the correct
  // size with respect to the current architecture
  if (LLVMType.isPointerTy()) {
    auto Size = model::Architecture::getPointerSize(Model.Architecture());
    return Size == ModelSize;
  }

  auto LLVMSize = LLVMType.getScalarSizeInBits();

  // Special case for i1
  if (LLVMSize < 8)
    return ModelSize == 1;

  return ModelSize * 8 == LLVMSize;
}

inline bool isAssignment(const llvm::Value *I) {
  return isCallToTagged(I, FunctionTags::Assign);
}

inline bool isLocalVarDecl(const llvm::Value *I) {
  return isCallToTagged(I, FunctionTags::LocalVariable);
}

inline bool isCallStackArgumentDecl(const llvm::Value *I) {
  auto *Call = dyn_cast_or_null<llvm::CallInst>(I);
  if (not Call)
    return false;

  auto *Callee = Call->getCalledFunction();
  if (not Callee)
    return false;

  return Callee->getName().startswith("revng_call_stack_arguments");
}

inline bool isArtificialAggregateLocalVarDecl(const llvm::Value *I) {
  return isCallToIsolatedFunction(I) and I->getType()->isAggregateType();
}

inline const llvm::CallInst *isCallToNonIsolated(const llvm::Value *I) {
  if (isCallToTagged(I, FunctionTags::QEMU)
      or isCallToTagged(I, FunctionTags::Helper)
      or isCallToTagged(I, FunctionTags::Exceptional)
      or llvm::isa<llvm::IntrinsicInst>(I))
    return llvm::cast<llvm::CallInst>(I);

  return nullptr;
}

inline bool isHelperAggregateLocalVarDecl(const llvm::Value *I) {
  return isCallToNonIsolated(I) and I->getType()->isAggregateType();
}

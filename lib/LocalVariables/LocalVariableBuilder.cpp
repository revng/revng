//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MathExtras.h"

#include "revng/ABI/ModelHelpers.h"
#include "revng/LocalVariables/LocalVariableBuilder.h"
#include "revng/LocalVariables/LocalVariableHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

using VB = LocalVariableBuilder<false>;

VariableBuilderTypes::VariableBuilderTypes(const model::Binary &TheBinary,
                                           llvm::Module &TheModule) :
  InputPointerSizedInteger{ getPointerSizedInteger(TheModule.getContext(),
                                                   TheBinary.Architecture()) },
  TargetPointerSizedInteger{
    TheModule.getDataLayout().getIntPtrType(TheModule.getContext())
  },
  Int8Ty{ llvm::Type::getInt8Ty(TheModule.getContext()) } {
}

VariableBuilderTypes::VariableBuilderTypes(llvm::Module &TheModule,
                                           unsigned InputPointerByteSize) :
  InputPointerSizedInteger{ llvm::IntegerType::get(TheModule.getContext(),
                                                   InputPointerByteSize * 8) },
  TargetPointerSizedInteger{
    TheModule.getDataLayout().getIntPtrType(TheModule.getContext())
  },
  Int8Ty{ llvm::Type::getInt8Ty(TheModule.getContext()) } {
}

/// Specialization of methods for non-legacy mode.
///
///@{

template<>
VB::LocalVarType *VB::createLocalVariable(const model::Type &VariableType) {
  size_t VariableSize = VariableType.size().value_or(0);
  revng_assert(VariableSize);

  revng::IRBuilder B(F->getContext());
  B.setInsertPointToFirstNonAlloca(*F);

  return B.CreateAlloca(llvm::ArrayType::get(Types.Int8Ty, VariableSize));
}

template<>
std::pair<VB::LocalVarType *, llvm::Instruction *>
VB::createLocalVariableAndTakeIntAddress(const model::Type &VariableType) {
  revng::IRBuilder B(F->getContext());
  B.setInsertPointToFirstNonAlloca(*F);
  auto *Variable = createLocalVariable(VariableType);
  return {
    Variable,
    cast<Instruction>(B.CreatePtrToInt(Variable,
                                       Types.InputPointerSizedInteger))
  };
}

template<>
Instruction *
VB::createCallStackArgumentVariable(const model::Type &VariableType) {
  return createLocalVariableAndTakeIntAddress(VariableType).second;
}

template<>
Instruction *VB::createStackFrameVariable(model::UpcastableType FrameType) {
  size_t StackSize = FrameType->size().value_or(0);
  revng_assert(StackSize);

  auto *ArrayType = ArrayType::get(Types.Int8Ty, StackSize);
  auto [AllocaStackFrame, PtrToInt] = createAllocaWithPtrToInt(F, ArrayType);
  setStackFrameMetadata(AllocaStackFrame);
  return cast<Instruction>(PtrToInt);
}

template<bool IsLegacy>
std::pair<llvm::AllocaInst *, llvm::Value *>
LocalVariableBuilder<IsLegacy>::createAllocaWithPtrToInt(llvm::Function *F,
                                                         llvm::Type *T) const {
  // TODO: try re-enabling checks here after dropping the old pipeline.
  revng::IRBuilder B(F->getContext());
  B.SetInsertPointPastAllocas(F);
  auto *Alloca = B.CreateAlloca(T);
  Value *PtrToInt = B.CreatePtrToInt(Alloca, Types.TargetPointerSizedInteger);

  if (Types.TargetPointerSizedInteger != Types.InputPointerSizedInteger) {
    // The target has a different bitsize than the input binary.
    // Inject an assumption about the pointer we built being representable in
    // the input bitsize to avoid LLVM emitting masks.
    auto InputBits = Types.InputPointerSizedInteger->getIntegerBitWidth();
    auto InputBitMask = maskTrailingOnes<uint64_t>(InputBits);
    B.CreateAssumption(B.CreateICmpEQ(B.CreateAnd(PtrToInt, InputBitMask),
                                      PtrToInt));
  }

  PtrToInt = B.CreateZExtOrTrunc(PtrToInt, Types.InputPointerSizedInteger);
  return { Alloca, PtrToInt };
}

///@}

// Instantiate specializations of LocalVariableBuilders
template class LocalVariableBuilder<false>;

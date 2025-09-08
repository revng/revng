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
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

// This name is not present in the emitted C.
RegisterIRHelper StackFrameMarker("revng_stack_frame");

// This name is not present in the emitted C.
RegisterIRHelper StackArgumentMarker("revng_call_stack_arguments");

template<bool IsLegacy>
using VarBuilder = LocalVariableBuilder<IsLegacy>;

using LegacyVB = VarBuilder</* IsLegacy */ true>;
using VB = VarBuilder</* IsLegacy */ false>;

static void addCommonAttributesAndTags(llvm::Function *F) {
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::WillReturn);
  // NoMerge, because merging two calls to one of these opcodes that
  // allocate local variable would mean merging the variables.
  F->addFnAttr(Attribute::NoMerge);
  F->setMemoryEffects(MemoryEffects::readOnly());
  F->setOnlyAccessesInaccessibleMemory();
  FunctionTags::AllocatesLocalVariable.addTo(F);
  FunctionTags::ReturnsPolymorphic.addTo(F);
  FunctionTags::IsRef.addTo(F);
}

/// Create and cache the function used to represent the allocation of the stack
/// frame
template<bool IsLegacy>
Function *VarBuilder<IsLegacy>::getStackFrameAllocator() {
  if constexpr (not IsLegacy)
    revng_abort("Only legacy LocalVariableBuilder should need stack frame "
                "allocator");

  if (nullptr == StackFrameAllocator) {
    auto *AllocatorType = FunctionType::get(InputPointerSizedInteger,
                                            { InputPointerSizedInteger },
                                            false);

    StackFrameAllocator = createIRHelper("revng_stack_frame",
                                         M,
                                         AllocatorType,
                                         GlobalValue::ExternalLinkage);
    addCommonAttributesAndTags(StackFrameAllocator);
  }

  return StackFrameAllocator;
}

/// Create and cache the function used to represent the allocation of the stack
/// arguments for calls to isolated functions.
template<bool IsLegacy>
Function *VarBuilder<IsLegacy>::getCallStackArgumentsAllocator() {
  if constexpr (not IsLegacy)
    revng_abort("Only legacy LocalVariableBuilder should need stack arguments "
                "allocator");

  if (nullptr == CallStackArgumentsAllocator) {
    llvm::Type *StringPtrType = getStringPtrType(M.getContext());
    auto *AllocatorType = FunctionType::get(InputPointerSizedInteger,
                                            { StringPtrType,
                                              InputPointerSizedInteger },
                                            false);

    CallStackArgumentsAllocator = createIRHelper("revng_call_stack_arguments",
                                                 M,
                                                 AllocatorType,
                                                 GlobalValue::ExternalLinkage);
    addCommonAttributesAndTags(CallStackArgumentsAllocator);
  }

  return CallStackArgumentsAllocator;
}

template<bool IsLegacy>
VarBuilder<IsLegacy>::LocalVariableBuilder(const model::Binary &TheBinary,
                                           Module &TheModule,
                                           OpaqueFunctionsPool<TypePair>
                                             *TheAddressOfPool) :
  Binary(TheBinary),
  M(TheModule),
  InputPointerSizedInteger(getPointerSizedInteger(M.getContext(), Binary)),
  TargetPointerSizedInteger(M.getDataLayout().getIntPtrType(M.getContext())),
  Int8Ty(llvm::Type::getInt8Ty(M.getContext())),
  F(nullptr),
  LocalVarPool(FunctionTags::LocalVariable.getPool(M)),
  AssignPool(FunctionTags::Assign.getPool(M)),
  CopyPool(FunctionTags::Copy.getPool(M)),
  AddressOfPool(*TheAddressOfPool) {

  // AddressOfPool and PtrSizedInteger should only be used when in Legacy mode.
  revng_assert(IsLegacy or (nullptr == TheAddressOfPool));
}

static const model::Function &getModelFunction(Function *F,
                                               const model::Binary &Binary) {
  MetaAddress Entry = getMetaAddressMetadata(F, "revng.function.entry");
  revng_assert(Entry.isValid());
  return Binary.Functions().at(Entry);
}

/// Specialization of methods for creating different kinds of local variables in
/// legacy mode.
///
/// TODO: drop these when we drop legacy mode
///@{

template<>
LegacyVB::LocalVarType *
LegacyVB::createLocalVariable(const model::Type &VariableType) {

  auto *LocalVarFunctionType = getLocalVarType(InputPointerSizedInteger);
  auto *LocalVarFunction = LocalVarPool.get(InputPointerSizedInteger,
                                            LocalVarFunctionType,
                                            "LocalVariable");

  revng::IRBuilder B(F->getContext());
  setInsertPointToFirstNonAlloca(B, *F);
  Constant *ReferenceString = toLLVMString(VariableType, M);
  return B.CreateCall(LocalVarFunction, { ReferenceString });
}

template<>
std::pair<LegacyVB::LocalVarType *, llvm::Instruction *>
LegacyVB::createLocalVariableAndTakeIntAddress(const model::Type
                                                 &VariableType) {

  LocalVarType *LocalVar = createLocalVariable(VariableType);

  revng::IRBuilder B(LocalVar->getNextNonDebugInstruction());

  // Take the address
  llvm::Type *T = LocalVar->getType();
  auto *AddressOfFunctionType = getAddressOfType(T, T);
  auto *AddressOfFunction = AddressOfPool.get({ T, T },
                                              AddressOfFunctionType,
                                              "AddressOf");
  Constant *ReferenceString = toLLVMString(VariableType, M);
  return { LocalVar,
           cast<Instruction>(B.CreateCall(AddressOfFunction,
                                          { ReferenceString, LocalVar })) };
}

template<>
Instruction *
LegacyVB::createCallStackArgumentVariable(const model::Type &VariableType) {
  size_t VariableSize = VariableType.size().value_or(0);
  revng_assert(VariableSize);

  llvm::Constant *VarTypeString = toLLVMString(VariableType, M);

  revng::IRBuilder B(F->getContext());
  setInsertPointToFirstNonAlloca(B, *F);

  Value *Size = ConstantInt::get(InputPointerSizedInteger, VariableSize);
  Instruction *Reference = B.CreateCall(getCallStackArgumentsAllocator(),
                                        { VarTypeString, Size });

  // Take the address
  llvm::Type *T = Reference->getType();
  auto *AddressOfFunctionType = getAddressOfType(T, T);
  auto *AddressOfFunction = AddressOfPool.get({ T, T },
                                              AddressOfFunctionType,
                                              "AddressOf");

  return cast<Instruction>(B.CreateCall(AddressOfFunction,
                                        { VarTypeString, Reference }));
}

template<>
Instruction *LegacyVB::createStackFrameVariable() {
  const model::Function &ModelFunction = getModelFunction(F, Binary);
  model::UpcastableType StackFrameType = ModelFunction.StackFrameType();

  size_t StackSize = StackFrameType->size().value_or(0);
  revng_assert(StackSize);

  revng::IRBuilder B(F->getContext());
  setInsertPointToFirstNonAlloca(B, *F);

  Instruction
    *Reference = B.CreateCall(getStackFrameAllocator(),
                              { ConstantInt::get(InputPointerSizedInteger,
                                                 StackSize) });
  // Take the address
  llvm::Type *T = Reference->getType();
  auto *AddressOfFunctionType = getAddressOfType(T, T);
  auto *AddressOfFunction = AddressOfPool.get({ T, T },
                                              AddressOfFunctionType,
                                              "AddressOf");

  llvm::Constant *StackTypeString = toLLVMString(StackFrameType, M);
  return cast<Instruction>(B.CreateCall(AddressOfFunction,
                                        { StackTypeString, Reference }));
}

template<>
LegacyVB::ReferenceType *
LegacyVB::getAssignedLocation(AssignType *Assign) const {
  revng_assert(isCallToTagged(Assign, FunctionTags::Assign));
  auto *AssignCall = getCallToTagged(Assign, FunctionTags::Assign);
  auto *LocalVariable = AssignCall->getArgOperand(1);
  revng_assert(isCallToTagged(LocalVariable, FunctionTags::LocalVariable)
               or isCallToTagged(LocalVariable, FunctionTags::IsRef));
  return cast<CallInst>(LocalVariable);
}

template<>
LegacyVB::CopyType *
LegacyVB::createCopyOnUse(ReferenceType *LocationToCopy, Use &U) {
  auto *InsertBefore = cast<Instruction>(U.getUser());
  revng::IRBuilder B(InsertBefore);

  // Create a Copy to dereference the LocalVariable
  auto *CopyFnType = getCopyType(U->getType(), LocationToCopy->getType());
  auto *CopyFunction = CopyPool.get(U->getType(), CopyFnType, "Copy");
  auto *Call = B.CreateCall(CopyFunction, { LocationToCopy });
  if (auto *InstructionLocation = dyn_cast<Instruction>(LocationToCopy))
    Call->setDebugLoc(InstructionLocation->getDebugLoc());
  return Call;
}

template<>
LegacyVB::CopyType *
LegacyVB::createCopyFromAssignedOnUse(AssignType *Assign, Use &U) {
  auto *AssignedLocation = getAssignedLocation(Assign);
  return createCopyOnUse(AssignedLocation, U);
}

template<>
LegacyVB::AssignType *
LegacyVB::createAssignmentBefore(Value *LocationToAssign,
                                 Value *ValueToAssign,
                                 Instruction *InsertBefore) {
  // Create an assignment that assigns ValueToAssign to LocationToAssign.
  revng::IRBuilder B(InsertBefore);
  auto *IRType = ValueToAssign->getType();
  auto *AssignFnType = getAssignFunctionType(IRType,
                                             LocationToAssign->getType());
  auto *AssignFunction = AssignPool.get(IRType, AssignFnType, "Assign");
  return B.CreateCall(AssignFunction, { ValueToAssign, LocationToAssign });
}

///@}

/// Specialization of methods for creating different kinds of local variables in
/// non-legacy mode.
///@{

template<>
VB::LocalVarType *VB::createLocalVariable(const model::Type &VariableType) {
  size_t VariableSize = VariableType.size().value_or(0);
  revng_assert(VariableSize);

  revng::IRBuilder B(F->getContext());
  setInsertPointToFirstNonAlloca(B, *F);

  // Create an alloca of array type with number of elements equal to
  // VariableSize (alloca [n x i8]), instead of creating an alloca of
  // VariableSize Int8Ty (alloca i8, n).
  // If we do the latter, LLVM's instcombine turns it into the former, but it
  // loses the variable type metadata that we need in the clifter.
  llvm::ArrayType *Array = llvm::ArrayType::get(Int8Ty, VariableSize);
  auto *AllocaLocalVariable = B.CreateAlloca(Array);
  setVariableTypeMetadata(AllocaLocalVariable, VariableType);
  return AllocaLocalVariable;
}

template<>
std::pair<VB::LocalVarType *, llvm::Instruction *>
VB::createLocalVariableAndTakeIntAddress(const model::Type &VariableType) {
  revng::IRBuilder B(F->getContext());
  setInsertPointToFirstNonAlloca(B, *F);
  auto *Variable = createLocalVariable(VariableType);
  return { Variable,
           cast<Instruction>(B.CreatePtrToInt(Variable,
                                              InputPointerSizedInteger)) };
}

template<>
Instruction *
VB::createCallStackArgumentVariable(const model::Type &VariableType) {
  return createLocalVariableAndTakeIntAddress(VariableType).second;
}

template<>
Instruction *VB::createStackFrameVariable() {
  const model::Function &ModelFunction = getModelFunction(F, Binary);
  model::UpcastableType StackFrameType = ModelFunction.StackFrameType();

  size_t StackSize = StackFrameType->size().value_or(0);
  revng_assert(StackSize);

  auto *ArrayType = ArrayType::get(Int8Ty, StackSize);
  auto [AllocaStackFrame, PtrToInt] = createAllocaWithPtrToInt(F, ArrayType);
  setStackTypeMetadata(AllocaStackFrame, *StackFrameType.get());
  return cast<Instruction>(PtrToInt);
}

template<>
VB::ReferenceType *VB::getAssignedLocation(AssignType *Assign) const {
  return cast<ReferenceType>(Assign->getPointerOperand());
}

template<>
VB::CopyType *VB::createCopyOnUse(ReferenceType *LocationToCopy, Use &U) {
  // Create a copy from the assigned location at the proper insertion point.
  auto *InsertBefore = cast<Instruction>(U.getUser());
  revng::IRBuilder B(InsertBefore);
  auto *Load = B.CreateLoad(U->getType(), LocationToCopy);
  if (auto *InstructionLocation = dyn_cast<Instruction>(LocationToCopy))
    Load->setDebugLoc(InstructionLocation->getDebugLoc());
  return Load;
}

template<>
VB::CopyType *VB::createCopyFromAssignedOnUse(AssignType *Assign, Use &U) {
  auto *AssignedLocation = cast<ReferenceType>(getAssignedLocation(Assign));
  return createCopyOnUse(AssignedLocation, U);
}

template<>
VB::AssignType *VB::createAssignmentBefore(Value *LocationToAssign,
                                           Value *ValueToAssign,
                                           Instruction *InsertBefore) {
  // Create a copy from the assigned location at the proper insertion point.
  revng::IRBuilder B(InsertBefore);
  if (auto *Instruction = llvm::dyn_cast<llvm::Instruction>(ValueToAssign))
    B.SetCurrentDebugLocation(Instruction->getDebugLoc());
  return B.CreateStore(ValueToAssign, LocationToAssign);
}

template<bool IsLegacy>
std::pair<llvm::AllocaInst *, llvm::Value *>
LocalVariableBuilder<IsLegacy>::createAllocaWithPtrToInt(llvm::Function *F,
                                                         llvm::Type *T) const {
  revng::IRBuilder B(M.getContext());
  B.SetInsertPointPastAllocas(F);
  B.SetCurrentDebugLocation(B.GetInsertPoint()->getDebugLoc());
  auto *Alloca = B.CreateAlloca(T);
  Value *PtrToInt = B.CreatePtrToInt(Alloca, TargetPointerSizedInteger);

  if (TargetPointerSizedInteger != InputPointerSizedInteger) {
    // The target has a different bitsize than the input binary.
    // Inject an assumption about the pointer we built being representable in
    // the input bitsize to avoid LLVM emitting masks.
    auto InputBits = InputPointerSizedInteger->getIntegerBitWidth();
    auto InputBitMask = maskTrailingOnes<uint64_t>(InputBits);
    B.CreateAssumption(B.CreateICmpEQ(B.CreateAnd(PtrToInt, InputBitMask),
                                      PtrToInt));
  }

  PtrToInt = B.CreateZExtOrTrunc(PtrToInt, InputPointerSizedInteger);
  return { Alloca, PtrToInt };
}

///@}

// Instantiate bosh specializations of LocalVariableBuilders and their
// constructors
template class LocalVariableBuilder<true>;
template class LocalVariableBuilder<false>;

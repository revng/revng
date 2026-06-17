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

LegacyCustomFunctions::LegacyCustomFunctions(llvm::Module &TheModule) :
  M(TheModule),
  LocalVarPool(FunctionTags::LocalVariable.getPool(M)),
  AssignPool(FunctionTags::Assign.getPool(M)),
  CopyPool(FunctionTags::Copy.getPool(M)),
  AddressOfPool(FunctionTags::AddressOf.getPool(M)) {
}

LegacyStackAllocators::LegacyStackAllocators(llvm::IntegerType
                                               *InputPointerSizedInteger,
                                             llvm::Module &TheModule) :
  M(TheModule) {
  auto *StackAllocatorType = FunctionType::get(InputPointerSizedInteger,
                                               { InputPointerSizedInteger },
                                               false);

  StackFrameAllocator = createIRHelper("revng_stack_frame",
                                       TheModule,
                                       StackAllocatorType,
                                       GlobalValue::ExternalLinkage);
  addCommonAttributesAndTags(StackFrameAllocator);

  llvm::Type *StringPtrType = getStringPtrType(TheModule.getContext());

  auto *AllocatorType = FunctionType::get(InputPointerSizedInteger,
                                          { StringPtrType,
                                            InputPointerSizedInteger },
                                          false);

  CallStackArgumentsAllocator = createIRHelper("revng_call_stack_arguments",
                                               TheModule,
                                               AllocatorType,
                                               GlobalValue::ExternalLinkage);
  addCommonAttributesAndTags(CallStackArgumentsAllocator);
}

LegacyStackAllocators
LegacyStackAllocators::makeLegacy(const model::Binary &TheBinary,
                                  llvm::Module &TheModule) {
  return LegacyStackAllocators(getPointerSizedInteger(TheModule.getContext(),
                                                      TheBinary.Architecture()),
                               TheModule);
}

/// Create and cache the function used to represent the allocation of the stack
/// frame
template<bool IsLegacy>
Function *VarBuilder<IsLegacy>::getStackFrameAllocator() {
  if constexpr (not IsLegacy)
    revng_abort("Only legacy LocalVariableBuilder should need stack frame "
                "allocator");

  return Allocators.value().StackFrameAllocator;
}

/// Create and cache the function used to represent the allocation of the stack
/// arguments for calls to isolated functions.
template<bool IsLegacy>
Function *VarBuilder<IsLegacy>::getCallStackArgumentsAllocator() {
  if constexpr (not IsLegacy)
    revng_abort("Only legacy LocalVariableBuilder should need stack arguments "
                "allocator");
  return Allocators.value().CallStackArgumentsAllocator;
}

/// Specialization of methods for legacy mode.
///
///@{

// TODO: drop these when we drop legacy mode

template<>
LegacyVB::LocalVarType *
LegacyVB::createLocalVariable(const model::Type &VariableType) {

  auto *LocalVarFunctionType = getLocalVarType(Types.InputPointerSizedInteger);
  auto *LocalVarFunction = CustomFunctions.value()
                             .LocalVarPool.get(Types.InputPointerSizedInteger,
                                               LocalVarFunctionType,
                                               "LocalVariable");

  // Here we should definitely use the builder that checks the debug info,
  // but since this going to go away soon, let it stay as is.
  revng::NonDebugInfoCheckingIRBuilder B(F->getContext());
  B.setInsertPointToFirstNonAlloca(*F);
  Constant *ReferenceString = toLLVMString(VariableType, *F->getParent());
  return B.CreateCall(LocalVarFunction, { ReferenceString });
}

template<>
std::pair<LegacyVB::LocalVarType *, llvm::Instruction *>
LegacyVB::createLocalVariableAndTakeIntAddress(const model::Type
                                                 &VariableType) {

  LocalVarType *LocalVar = createLocalVariable(VariableType);

  // Here we should definitely use the builder that checks the debug info,
  // but since this going to go away soon, let it stay as is.
  revng::NonDebugInfoCheckingIRBuilder // formatting
    B(LocalVar->getNextNonDebugInstruction());

  // Take the address
  llvm::Type *T = LocalVar->getType();
  auto *AddressOfFunctionType = getAddressOfType(T, T);
  auto *AddressOfFunction = CustomFunctions.value()
                              .AddressOfPool.get({ T, T },
                                                 AddressOfFunctionType,
                                                 "AddressOf");
  Constant *ReferenceString = toLLVMString(VariableType, *F->getParent());
  return { LocalVar,
           cast<Instruction>(B.CreateCall(AddressOfFunction,
                                          { ReferenceString, LocalVar })) };
}

template<>
Instruction *
LegacyVB::createCallStackArgumentVariable(const model::Type &VariableType) {
  size_t VariableSize = VariableType.size().value_or(0);
  revng_assert(VariableSize);

  llvm::Constant *VarTypeString = toLLVMString(VariableType, *F->getParent());

  revng::NonDebugInfoCheckingIRBuilder B(F->getContext());
  B.setInsertPointToFirstNonAlloca(*F);

  Value *Size = ConstantInt::get(Types.InputPointerSizedInteger, VariableSize);
  Instruction *Reference = B.CreateCall(getCallStackArgumentsAllocator(),
                                        { VarTypeString, Size });

  // Take the address
  llvm::Type *T = Reference->getType();
  auto *AddressOfFunctionType = getAddressOfType(T, T);
  auto *AddressOfFunction = CustomFunctions.value()
                              .AddressOfPool.get({ T, T },
                                                 AddressOfFunctionType,
                                                 "AddressOf");

  return cast<Instruction>(B.CreateCall(AddressOfFunction,
                                        { VarTypeString, Reference }));
}

template<>
Instruction *
LegacyVB::createStackFrameVariable(model::UpcastableType FrameType) {
  size_t StackSize = FrameType->size().value_or(0);
  revng_assert(StackSize);

  revng::NonDebugInfoCheckingIRBuilder B(F->getContext());
  B.setInsertPointToFirstNonAlloca(*F);

  Instruction
    *Reference = B.CreateCall(getStackFrameAllocator(),
                              { ConstantInt::get(Types.InputPointerSizedInteger,
                                                 StackSize) });
  // Take the address
  llvm::Type *T = Reference->getType();
  auto *AddressOfFunctionType = getAddressOfType(T, T);
  auto *AddressOfFunction = CustomFunctions.value()
                              .AddressOfPool.get({ T, T },
                                                 AddressOfFunctionType,
                                                 "AddressOf");

  llvm::Constant *StackTypeString = toLLVMString(FrameType, *F->getParent());
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
  auto *InsertBefore = llvm::cast<llvm::Instruction>(U.getUser());
  llvm::DebugLoc DebugLocation = InsertBefore->getDebugLoc();
  if (auto *Instruction = llvm::dyn_cast<llvm::Instruction>(LocationToCopy))
    DebugLocation = Instruction->getDebugLoc();

  revng::NonDebugInfoCheckingIRBuilder B(InsertBefore, DebugLocation);

  // Create a Copy to dereference the LocalVariable
  auto *CopyFnType = getCopyType(U->getType(), LocationToCopy->getType());
  auto *CopyFunction = CustomFunctions.value().CopyPool.get(U->getType(),
                                                            CopyFnType,
                                                            "Copy");
  return B.CreateCall(CopyFunction, { LocationToCopy });
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
  llvm::DebugLoc DebugLocation = InsertBefore->getDebugLoc();
  if (auto *Instruction = llvm::dyn_cast<llvm::Instruction>(ValueToAssign))
    DebugLocation = Instruction->getDebugLoc();

  // Create an assignment that assigns ValueToAssign to LocationToAssign.
  revng::NonDebugInfoCheckingIRBuilder B(InsertBefore, DebugLocation);
  auto *IRType = ValueToAssign->getType();
  auto *AssignFnType = getAssignFunctionType(IRType,
                                             LocationToAssign->getType());
  auto *AssignFunction = CustomFunctions.value().AssignPool.get(IRType,
                                                                AssignFnType,
                                                                "Assign");
  return B.CreateCall(AssignFunction, { ValueToAssign, LocationToAssign });
}

///@}

/// Specialization of methods for non-legacy mode.
///
///@{

template<>
VB::LocalVarType *VB::createLocalVariable(const model::Type &VariableType) {
  size_t VariableSize = VariableType.size().value_or(0);
  revng_assert(VariableSize);

  revng::NonDebugInfoCheckingIRBuilder B(F->getContext());
  B.setInsertPointToFirstNonAlloca(*F);

  return B.CreateAlloca(llvm::ArrayType::get(Types.Int8Ty, VariableSize));
}

template<>
std::pair<VB::LocalVarType *, llvm::Instruction *>
VB::createLocalVariableAndTakeIntAddress(const model::Type &VariableType) {
  revng::NonDebugInfoCheckingIRBuilder B(F->getContext());
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

template<>
VB::ReferenceType *VB::getAssignedLocation(AssignType *Assign) const {
  revng_abort("deprecated: should never be called in non-legacy mode");
  return nullptr;
}

template<>
VB::CopyType *VB::createCopyOnUse(ReferenceType *LocationToCopy, Use &U) {
  revng_abort("deprecated: should never be called in non-legacy mode");
  return nullptr;
}

template<>
VB::CopyType *VB::createCopyFromAssignedOnUse(AssignType *Assign, Use &U) {
  revng_abort("deprecated: should never be called in non-legacy mode");
  return nullptr;
}

template<>
VB::AssignType *VB::createAssignmentBefore(Value *LocationToAssign,
                                           Value *ValueToAssign,
                                           Instruction *InsertBefore) {
  revng_abort("deprecated: should never be called in non-legacy mode");
  return nullptr;
}

template<bool IsLegacy>
std::pair<llvm::AllocaInst *, llvm::Value *>
VarBuilder<IsLegacy>::createAllocaWithPtrToInt(llvm::Function *F,
                                               llvm::Type *T) const {
  // TODO: try re-enabling checks here after dropping the old pipeline.
  revng::NonDebugInfoCheckingIRBuilder B(F->getContext());
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
template class LocalVariableBuilder<true>;
template class LocalVariableBuilder<false>;

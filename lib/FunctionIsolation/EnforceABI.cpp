/// \file EnforceABI.cpp
/// \brief Promotes global variables CSV to function arguments or local
///        variables, according to the ABI analysis.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/ADT/LazySmallBitVector.h"
#include "revng/ADT/SmallMap.h"
#include "revng/EarlyFunctionAnalysis/ABI.h"
#include "revng/FunctionIsolation/EnforceABI.h"
#include "revng/FunctionIsolation/StructInitializers.h"
#include "revng/Model/Register.h"
#include "revng/Model/Type.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/OpaqueFunctionsPool.h"

using namespace llvm;

char EnforceABI::ID = 0;
using Register = RegisterPass<EnforceABI>;
static Register X("enforce-abi", "Enforce ABI Pass", true, true);

static Logger<> EnforceABILog("enforce-abi");

class EnforceABIImpl {
public:
  EnforceABIImpl(Module &M,
                 GeneratedCodeBasicInfo &GCBI,
                 model::Binary &Binary) :
    M(M),
    GCBI(GCBI),
    FunctionDispatcher(M.getFunction("function_dispatcher")),
    Context(M.getContext()),
    Initializers(&M),
    Binary(Binary),
    MetaAddressStruct(MetaAddress::getStruct(&M)) {}

  void run();

private:
  Function *
  handleFunction(Function &OldFunction, const model::Function &FunctionModel);
  Function *recreateFunction(Function &OldFunction,
                             const model::RawFunctionType &Prototype);
  void
  createPrologue(Function *NewFunction, const model::Function &FunctionModel);

  void handleRegularFunctionCall(CallInst *Call);
  void generateCall(IRBuilder<> &Builder,
                    FunctionCallee Callee,
                    const model::BasicBlock &CallSiteBlock,
                    const model::CallEdge &CallSite);

private:
  Module &M;
  GeneratedCodeBasicInfo &GCBI;
  std::map<Function *, const model::Function *> FunctionsMap;
  std::map<Function *, Function *> OldToNew;
  Function *FunctionDispatcher;
  Function *OpaquePC;
  LLVMContext &Context;
  StructInitializers Initializers;
  model::Binary &Binary;
  StructType *MetaAddressStruct;
};

bool EnforceABI::runOnModule(Module &M) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  // TODO: prepopulate type system with basic types of the ABI, so this can be
  //       const
  model::Binary &Binary = *ModelWrapper.getWriteableModel().get();

  EnforceABIImpl Impl(M, GCBI, Binary);
  Impl.run();
  return false;
}

void EnforceABIImpl::run() {
  // Declare an opaque function used later to obtain a value to store in the
  // local %pc alloca, so that we don't incur in error when removing the bad
  // return pc checks.
  Type *PCType = GCBI.pcReg()->getType()->getPointerElementType();
  auto *OpaqueFT = FunctionType::get(PCType, {}, false);
  OpaquePC = Function::Create(OpaqueFT,
                              Function::ExternalLinkage,
                              "opaque_pc",
                              M);
  OpaquePC->addFnAttr(Attribute::NoUnwind);
  OpaquePC->addFnAttr(Attribute::WillReturn);
  OpaquePC->addFnAttr(Attribute::ReadOnly);
  FunctionTags::OpaqueCSVValue.addTo(OpaquePC);

  std::vector<Function *> OldFunctions;
  if (FunctionDispatcher != nullptr)
    OldFunctions.push_back(FunctionDispatcher);

  // Recreate dynamic functions with arguments
  for (const model::DynamicFunction &FunctionModel :
       Binary.ImportedDynamicFunctions) {
    // TODO: have an API to go from model to llvm::Function
    auto OldFunctionName = (Twine("dynamic_") + FunctionModel.name()).str();
    Function *OldFunction = M.getFunction(OldFunctionName);
    revng_assert(OldFunction != nullptr);
    OldFunctions.push_back(OldFunction);

    const auto *Type = FunctionModel.Prototype.get();
    revng_assert(Type != nullptr);

    auto Prototype = abi::getRawFunctionTypeOrDefault(Binary, Type);
    Function *NewFunction = recreateFunction(*OldFunction, Prototype);
    FunctionTags::DynamicFunction.addTo(NewFunction);

    // EnforceABI currently does not support execution
    NewFunction->deleteBody();

    OldToNew[OldFunction] = NewFunction;
  }

  // Recreate isolated functions with arguments
  for (const model::Function &FunctionModel : Binary.Functions) {
    if (FunctionModel.Type == model::FunctionType::Fake)
      continue;

    revng_assert(not FunctionModel.name().empty());
    auto OldFunctionName = (Twine("local_") + FunctionModel.name()).str();
    Function *OldFunction = M.getFunction(OldFunctionName);
    revng_assert(OldFunction != nullptr);
    OldFunctions.push_back(OldFunction);

    Function *NewFunction = handleFunction(*OldFunction, FunctionModel);

    FunctionsMap[NewFunction] = &FunctionModel;
    OldToNew[OldFunction] = NewFunction;
  }

  auto IsInIsolatedFunction = [this](Instruction *I) -> bool {
    return FunctionsMap.count(I->getParent()->getParent()) != 0;
  };

  // Handle function calls in isolated functions
  std::vector<CallInst *> RegularCalls;
  for (auto *F : OldFunctions)
    for (User *U : F->users())
      if (auto *Call = dyn_cast<CallInst>(skipCasts(U)))
        if (IsInIsolatedFunction(Call))
          RegularCalls.push_back(Call);

  for (CallInst *Call : RegularCalls)
    handleRegularFunctionCall(Call);

  // Drop all the old functions, after we stole all of its blocks
  for (Function *OldFunction : OldFunctions)
    OldFunction->eraseFromParent();

  // Quick and dirty DCE
  for (auto [F, _] : FunctionsMap)
    EliminateUnreachableBlocks(*F, nullptr, false);

  if (VerifyLog.isEnabled()) {
    raw_os_ostream Stream(dbg);
    revng_assert(not verifyModule(M, &Stream));
  }
}

static Type *getLLVMTypeForRegister(Module *M, model::Register::Values V) {
  LLVMContext &C = M->getContext();
  return IntegerType::getIntNTy(C, 8 * model::Register::getSize(V));
}

static std::pair<Type *, SmallVector<Type *, 8>>
getLLVMReturnTypeAndArguments(llvm::Module *M,
                              const model::RawFunctionType &Prototype) {
  using model::NamedTypedRegister;
  using model::RawFunctionType;
  using model::TypedRegister;

  LLVMContext &Context = M->getContext();

  SmallVector<llvm::Type *, 8> ArgumentsTypes;
  SmallVector<llvm::Type *, 8> ReturnTypes;

  for (const NamedTypedRegister &TR : Prototype.Arguments)
    ArgumentsTypes.push_back(getLLVMTypeForRegister(M, TR.Location));

  for (const TypedRegister &TR : Prototype.ReturnValues)
    ReturnTypes.push_back(getLLVMTypeForRegister(M, TR.Location));

  // Create the return type
  Type *ReturnType = Type::getVoidTy(Context);
  if (ReturnTypes.size() == 0)
    ReturnType = Type::getVoidTy(Context);
  else if (ReturnTypes.size() == 1)
    ReturnType = ReturnTypes[0];
  else
    ReturnType = StructType::create(ReturnTypes);

  // Create new function
  return { ReturnType, ArgumentsTypes };
}

static FunctionType *
toLLVMType(llvm::Module *M, const model::RawFunctionType &Prototype) {
  auto [ReturnType, Arguments] = getLLVMReturnTypeAndArguments(M, Prototype);
  return FunctionType::get(ReturnType, Arguments, false);
}

Function *EnforceABIImpl::handleFunction(Function &OldFunction,
                                         const model::Function &FunctionModel) {
  const model::Type *PrototypeType = FunctionModel.Prototype.get();
  const auto &Prototype = *cast<model::RawFunctionType>(PrototypeType);
  Function *NewFunction = recreateFunction(OldFunction, Prototype);
  FunctionTags::Lifted.addTo(NewFunction);
  createPrologue(NewFunction, FunctionModel);
  return NewFunction;
}

Function *
EnforceABIImpl::recreateFunction(Function &OldFunction,
                                 const model::RawFunctionType &Prototype) {
  // Create new function
  auto [NewReturnType, NewArguments] = getLLVMReturnTypeAndArguments(&M,
                                                                     Prototype);
  auto *NewFunction = changeFunctionType(OldFunction,
                                         NewReturnType,
                                         NewArguments);

  // Set argument names
  for (const auto &[LLVMArgument, ModelArgument] :
       zip(NewFunction->args(), Prototype.Arguments))
    LLVMArgument.setName(ModelArgument.name());

  return NewFunction;
}

void EnforceABIImpl::createPrologue(Function *NewFunction,
                                    const model::Function &FunctionModel) {
  using model::NamedTypedRegister;
  using model::RawFunctionType;
  using model::TypedRegister;

  const auto &Prototype = *cast<RawFunctionType>(FunctionModel.Prototype.get());

  SmallVector<GlobalVariable *, 8> ArgumentCSVs;
  SmallVector<GlobalVariable *, 8> ReturnCSVs;

  // We sort arguments by their CSV name
  for (const NamedTypedRegister &TR : Prototype.Arguments) {
    auto Name = ABIRegister::toCSVName(TR.Location);
    auto *CSV = cast<GlobalVariable>(M.getGlobalVariable(Name, true));
    ArgumentCSVs.push_back(CSV);
  }

  for (const TypedRegister &TR : Prototype.ReturnValues) {
    auto Name = ABIRegister::toCSVName(TR.Location);
    auto *CSV = cast<GlobalVariable>(M.getGlobalVariable(Name, true));
    ReturnCSVs.push_back(CSV);
  }

  // Store arguments to CSVs
  BasicBlock &Entry = NewFunction->getEntryBlock();
  IRBuilder<> StoreBuilder(Entry.getTerminator());
  for (const auto &[TheArgument, CSV] : zip(NewFunction->args(), ArgumentCSVs))
    StoreBuilder.CreateStore(&TheArgument, CSV);

  // Build the return value
  if (ReturnCSVs.size() != 0) {
    for (BasicBlock &BB : *NewFunction) {
      if (auto *Return = dyn_cast<ReturnInst>(BB.getTerminator())) {
        IRBuilder<> Builder(Return);
        std::vector<Value *> ReturnValues;
        for (GlobalVariable *ReturnCSV : ReturnCSVs)
          ReturnValues.push_back(Builder.CreateLoad(ReturnCSV));

        if (ReturnValues.size() == 1)
          Builder.CreateRet(ReturnValues[0]);
        else
          Initializers.createReturn(Builder, ReturnValues);

        Return->eraseFromParent();
      }
    }
  }
}

void EnforceABIImpl::handleRegularFunctionCall(CallInst *Call) {
  Function *Caller = Call->getParent()->getParent();
  const model::Function &FunctionModel = *FunctionsMap.at(Caller);

  Function *CallerFunction = Call->getParent()->getParent();

  Function *Callee = cast<Function>(skipCasts(Call->getCalledOperand()));
  bool IsDirect = (Callee != FunctionDispatcher);
  if (IsDirect)
    Callee = OldToNew.at(Callee);

  // Identify the corresponding call site in the model
  MetaAddress BasicBlockAddress = GCBI.getJumpTarget(Call->getParent());
  const model::BasicBlock &Block = FunctionModel.CFG.at(BasicBlockAddress);
  const model::CallEdge *CallSite = nullptr;
  for (const auto &Edge : Block.Successors) {
    using namespace model::FunctionEdgeType;
    CallSite = dyn_cast<model::CallEdge>(Edge.get());
    if (CallSite != nullptr)
      break;
  }

  // Note that currently, in case of indirect call, we emit a call to a
  // placeholder function that will throw an exception. If exceptions are
  // correctly supported post enforce-abi, and the ABI data is correct, this
  // should work. However this is not very efficient.
  //
  // Alternatives:
  //
  // 1. Emit an inline dispatcher that calls all the compatible functions (i.e.,
  //    they take a subset of the call site's arguments and return a superset of
  //    the call site's return values).
  // 2. We have a dedicated outlined dispatcher that takes all the arguments of
  //    the call site, plus all the registers of the return values. Under the
  //    assumption that each return value of the call site is either a return
  //    value of the callee or is preserved by the callee, we can fill each
  //    return value using the callee's return value or the argument
  //    representing the value of that register before the call.
  //    In case the call site expects a return value that is neither a return
  //    value nor a preserved register or the callee, we exclude it from the
  ///   switch.

  // Generate the call
  IRBuilder<> Builder(Call);
  generateCall(Builder, Callee, Block, *CallSite);

  // Create an additional store to the local %pc, so that the optimizer cannot
  // do stuff with llvm.assume.
  revng_assert(OpaquePC != nullptr);
  Builder.CreateStore(Builder.CreateCall(OpaquePC), GCBI.pcReg());

  // Drop the original call
  Call->eraseFromParent();
}

static FunctionCallee
toFunctionPointer(IRBuilder<> &B, Value *V, FunctionType *FT) {
  Module *M = getModule(B.GetInsertBlock());
  const auto &DL = M->getDataLayout();
  IntegerType *IntPtrTy = DL.getIntPtrType(M->getContext());
  Value *Callee = B.CreateIntToPtr(V, FT->getPointerTo());
  return FunctionCallee(FT, Callee);
}

void EnforceABIImpl::generateCall(IRBuilder<> &Builder,
                                  FunctionCallee Callee,
                                  const model::BasicBlock &CallSiteBlock,
                                  const model::CallEdge &CallSite) {
  using model::NamedTypedRegister;
  using model::RawFunctionType;
  using model::TypedRegister;

  revng_assert(Callee.getCallee() != nullptr);

  llvm::SmallVector<Value *, 8> Arguments;
  llvm::SmallVector<GlobalVariable *, 8> ReturnCSVs;

  model::TypePath PrototypePath = getPrototype(Binary, CallSite);
  const auto &Prototype = abi::getRawFunctionTypeOrDefault(Binary,
                                                           PrototypePath.get());

  bool IsIndirect = (Callee.getCallee() == FunctionDispatcher);
  if (IsIndirect) {
    // Create a new `indirect_placeholder` function with the specific function
    // type we need
    Value *PC = GCBI.programCounterHandler()->loadJumpablePC(Builder);
    auto *NewType = toLLVMType(&M, Prototype);
    Callee = toFunctionPointer(Builder, PC, NewType);
  } else {
    BasicBlock *InsertBlock = Builder.GetInsertPoint()->getParent();
    revng_log(EnforceABILog,
              "Emitting call to " << getName(Callee.getCallee()) << " from "
                                  << getName(InsertBlock));
  }

  //
  // Collect arguments and returns
  //
  for (const NamedTypedRegister &TR : Prototype.Arguments) {
    auto Name = ABIRegister::toCSVName(TR.Location);
    GlobalVariable *CSV = M.getGlobalVariable(Name, true);
    Arguments.push_back(Builder.CreateLoad(CSV));
  }

  for (const TypedRegister &TR : Prototype.ReturnValues) {
    auto Name = ABIRegister::toCSVName(TR.Location);
    GlobalVariable *CSV = M.getGlobalVariable(Name, true);
    ReturnCSVs.push_back(CSV);
  }

  //
  // Produce the call
  //
  auto *Result = Builder.CreateCall(Callee, Arguments);
  GCBI.setMetaAddressMetadata(Result,
                              "revng.callerblock.start",
                              CallSiteBlock.Start);
  if (ReturnCSVs.size() != 1) {
    unsigned I = 0;
    for (GlobalVariable *ReturnCSV : ReturnCSVs) {
      Builder.CreateStore(Builder.CreateExtractValue(Result, { I }), ReturnCSV);
      I++;
    }
  } else {
    Builder.CreateStore(Result, ReturnCSVs[0]);
  }
}

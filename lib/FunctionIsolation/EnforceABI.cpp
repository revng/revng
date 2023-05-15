/// \file EnforceABI.cpp
/// \brief Promotes global variables CSV to function arguments or local
///        variables, according to the ABI analysis.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/LazySmallBitVector.h"
#include "revng/ADT/SmallMap.h"
#include "revng/EarlyFunctionAnalysis/CallEdge.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/FunctionIsolation/EnforceABI.h"
#include "revng/FunctionIsolation/StructInitializers.h"
#include "revng/Model/Register.h"
#include "revng/Model/Type.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/OpaqueFunctionsPool.h"

using namespace llvm;
using FTLayout = abi::FunctionType::Layout;

char EnforceABI::ID = 0;
using Register = RegisterPass<EnforceABI>;
static Register X("enforce-abi", "Enforce ABI Pass", true, true);

static Logger<> EnforceABILog("enforce-abi");

struct EnforceABIPipe {
  static constexpr auto Name = "enforce-abi";

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace pipeline;
    using namespace ::revng::kinds;
    return { ContractGroup::transformOnlyArgument(Isolated,
                                                  ABIEnforced,
                                                  InputPreservation::Erase) };
  }

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new EnforceABI());
  }
};

static pipeline::RegisterLLVMPass<EnforceABIPipe> Y;

class EnforceABIImpl {
public:
  EnforceABIImpl(Module &M,
                 GeneratedCodeBasicInfo &GCBI,
                 const model::Binary &Binary,
                 FunctionMetadataCache &Cache) :
    M(M),
    GCBI(GCBI),
    FunctionDispatcher(M.getFunction("function_dispatcher")),
    Context(M.getContext()),
    Initializers(&M),
    Binary(Binary),
    Cache(&Cache) {}

  void run();

private:
  Function *handleFunction(Function &OldFunction,
                           const model::Function &FunctionModel);

  Function *recreateFunction(Function &OldFunction, const FTLayout &Prototype);

  void createPrologue(Function *NewFunction,
                      const model::Function &FunctionModel);

  void handleRegularFunctionCall(CallInst *Call);
  CallInst *generateCall(IRBuilder<> &Builder,
                         MetaAddress Entry,
                         FunctionCallee Callee,
                         const efa::BasicBlock &CallSiteBlock,
                         const efa::CallEdge &CallSite);

private:
  Module &M;
  GeneratedCodeBasicInfo &GCBI;
  std::map<Function *, const model::Function *> FunctionsMap;
  std::map<Function *, Function *> OldToNew;
  Function *FunctionDispatcher;
  LLVMContext &Context;
  StructInitializers Initializers;
  const model::Binary &Binary;
  FunctionMetadataCache *Cache;
};

bool EnforceABI::runOnModule(Module &M) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  // TODO: prepopulate type system with basic types of the ABI, so this can be
  //       const
  const model::Binary &Binary = *ModelWrapper.getReadOnlyModel().get();

  EnforceABIImpl Impl(M,
                      GCBI,
                      Binary,
                      getAnalysis<FunctionMetadataCachePass>().get());
  Impl.run();
  return false;
}

void EnforceABIImpl::run() {
  std::vector<Function *> OldFunctions;
  if (FunctionDispatcher != nullptr)
    OldFunctions.push_back(FunctionDispatcher);

  // Recreate dynamic functions with arguments
  for (const model::DynamicFunction &FunctionModel :
       Binary.ImportedDynamicFunctions()) {
    // TODO: have an API to go from model to llvm::Function
    auto OldFunctionName = (Twine("dynamic_") + FunctionModel.OriginalName())
                             .str();
    Function *OldFunction = M.getFunction(OldFunctionName);
    if (not OldFunction or OldFunction->isDeclaration())
      continue;
    OldFunctions.push_back(OldFunction);

    using namespace abi::FunctionType;
    auto Prototype = Layout::make(FunctionModel.prototype(Binary));
    revng_assert(Prototype.verify());

    Function *NewFunction = recreateFunction(*OldFunction, Prototype);

    // EnforceABI currently does not support execution
    NewFunction->deleteBody();

    // Copy metadata
    NewFunction->copyMetadata(OldFunction, 0);

    OldToNew[OldFunction] = NewFunction;
  }

  // Recreate isolated functions with arguments
  for (const model::Function &FunctionModel : Binary.Functions()) {
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
    eraseFromParent(OldFunction);

  // Quick and dirty DCE
  for (auto [F, _] : FunctionsMap)
    if (not F->isDeclaration())
      EliminateUnreachableBlocks(*F, nullptr, false);

  revng::verify(&M);
}

static Type *getLLVMTypeForRegister(Module *M, model::Register::Values V) {
  LLVMContext &C = M->getContext();
  return IntegerType::getIntNTy(C, 8 * model::Register::getSize(V));
}

static std::pair<Type *, SmallVector<Type *, 8>>
getLLVMReturnTypeAndArguments(llvm::Module *M, const FTLayout &Prototype) {
  using model::NamedTypedRegister;
  using model::RawFunctionType;
  using model::TypedRegister;

  LLVMContext &Context = M->getContext();

  SmallVector<llvm::Type *, 8> ArgumentsTypes;
  SmallVector<llvm::Type *, 8> ReturnTypes;

  for (model::Register::Values Register : Prototype.argumentRegisters())
    ArgumentsTypes.push_back(getLLVMTypeForRegister(M, Register));

  for (model::Register::Values Register : Prototype.returnValueRegisters())
    ReturnTypes.push_back(getLLVMTypeForRegister(M, Register));

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

Function *EnforceABIImpl::handleFunction(Function &OldFunction,
                                         const model::Function &FunctionModel) {
  bool IsDeclaration = OldFunction.isDeclaration();

  auto Prototype = FunctionModel.prototype(Binary);
  auto Layout = abi::FunctionType::Layout::make(Prototype);
  Function *NewFunction = recreateFunction(OldFunction, Layout);

  FunctionTags::ABIEnforced.addTo(NewFunction);

  if (not IsDeclaration)
    createPrologue(NewFunction, FunctionModel);

  return NewFunction;
}

Function *EnforceABIImpl::recreateFunction(Function &OldFunction,
                                           const FTLayout &Prototype) {
  // Create new function
  auto [NewReturnType, NewArguments] = getLLVMReturnTypeAndArguments(&M,
                                                                     Prototype);
  auto *NewFunction = changeFunctionType(OldFunction,
                                         NewReturnType,
                                         NewArguments);

  revng_assert(NewFunction->arg_size() == Prototype.argumentRegisterCount());
  for (size_t Index = 0; const auto &Argument : Prototype.Arguments)
    for (model::Register::Values Register : Argument.Registers)
      NewFunction->getArg(Index++)->setName(model::Register::getName(Register));

  return NewFunction;
}

static GlobalVariable *tryGetCSV(Module *M, model::Register::Values Register) {
  auto Name = model::Register::getCSVName(Register);
  return M->getGlobalVariable(Name, true);
}

static Value *loadCSVOrUndef(IRBuilder<> &Builder,
                             Module *M,
                             model::Register::Values Register) {
  GlobalVariable *CSV = tryGetCSV(M, Register);
  if (CSV == nullptr) {
    auto Size = model::Register::getSize(Register);
    auto *Type = IntegerType::get(M->getContext(), Size * 8);
    return UndefValue::get(Type);
  } else {
    return createLoad(Builder, CSV);
  }
}

static std::pair<Type *, Constant *>
getCSVOrUndef(Module *M, model::Register::Values Register) {
  GlobalVariable *CSV = tryGetCSV(M, Register);
  if (CSV == nullptr) {
    auto Size = model::Register::getSize(Register);
    auto *Type = IntegerType::get(M->getContext(), Size * 8);
    return { Type, UndefValue::get(PointerType::get(M->getContext(), 0)) };
  } else {
    return { CSV->getValueType(), CSV };
  }
}

void EnforceABIImpl::createPrologue(Function *NewFunction,
                                    const model::Function &FunctionModel) {
  using model::NamedTypedRegister;
  using model::RawFunctionType;
  using model::TypedRegister;

  auto Prototype = FTLayout::make(FunctionModel.prototype(Binary));

  SmallVector<Constant *, 8> ArgumentCSVs;
  SmallVector<std::pair<Type *, Constant *>, 8> ReturnCSVs;

  // We sort arguments by their CSV name
  for (model::Register::Values Register : Prototype.argumentRegisters())
    ArgumentCSVs.push_back(getCSVOrUndef(&M, Register).second);

  for (model::Register::Values Register : Prototype.returnValueRegisters())
    ReturnCSVs.push_back(getCSVOrUndef(&M, Register));

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
        for (auto [Type, ReturnCSV] : ReturnCSVs)
          ReturnValues.push_back(Builder.CreateLoad(Type, ReturnCSV));

        if (ReturnValues.size() == 1)
          Builder.CreateRet(ReturnValues[0]);
        else
          Initializers.createReturn(Builder, ReturnValues);

        eraseFromParent(Return);
      }
    }
  }
}

void EnforceABIImpl::handleRegularFunctionCall(CallInst *Call) {
  revng_assert(Call->getDebugLoc());
  Function *Caller = Call->getParent()->getParent();
  const model::Function &FunctionModel = *FunctionsMap.at(Caller);

  Function *CallerFunction = Call->getParent()->getParent();

  Function *Callee = cast<Function>(skipCasts(Call->getCalledOperand()));
  bool IsDirect = (Callee != FunctionDispatcher);
  if (IsDirect)
    Callee = OldToNew.at(Callee);

  // Identify the corresponding call site in the model
  const efa::FunctionMetadata &FM = Cache->getFunctionMetadata(CallerFunction);

  const efa::BasicBlock *CallerBlock = FM.findBlock(GCBI, Call->getParent());
  revng_assert(CallerBlock != nullptr);

  // Find the CallEdge
  const efa::CallEdge *CallSite = nullptr;
  for (const auto &Edge : CallerBlock->Successors()) {
    using namespace efa::FunctionEdgeType;
    CallSite = dyn_cast<efa::CallEdge>(Edge.get());
    if (CallSite != nullptr)
      break;
  }
  revng_assert(CallSite != nullptr);

  // Note that currently, in case of indirect call, we emit a call to a
  // placeholder function that will throw an exception. If exceptions are
  // correctly supported post enforce-abi, and the ABI data is correct, this
  // should work. However this is not very efficient.
  //
  // Alternatives:
  //
  // 1. Emit an inline dispatcher that calls all the compatible function (i.e.,
  //    they take a subset of the call site's arguments and return a superset
  //    of the call site's return values).
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
  CallInst *NewCall = generateCall(Builder,
                                   FunctionModel.Entry(),
                                   Callee,
                                   *CallerBlock,
                                   *CallSite);
  NewCall->copyMetadata(*Call);

  // Set PC to the expected value
  GCBI.programCounterHandler()->setPC(Builder, CallerBlock->ID().start());

  // Drop the original call
  eraseFromParent(Call);
}

static FunctionCallee
toFunctionPointer(IRBuilder<> &B, Value *V, FunctionType *FT) {
  Module *M = getModule(B.GetInsertBlock());
  const auto &DL = M->getDataLayout();
  IntegerType *IntPtrTy = DL.getIntPtrType(M->getContext());
  Value *Callee = B.CreateIntToPtr(V, FT->getPointerTo());
  return FunctionCallee(FT, Callee);
}

CallInst *EnforceABIImpl::generateCall(IRBuilder<> &Builder,
                                       MetaAddress Entry,
                                       FunctionCallee Callee,
                                       const efa::BasicBlock &CallSiteBlock,
                                       const efa::CallEdge &CallSite) {
  using model::NamedTypedRegister;
  using model::RawFunctionType;
  using model::TypedRegister;

  revng_assert(Callee.getCallee() != nullptr);

  llvm::SmallVector<Value *, 8> Arguments;
  llvm::SmallVector<Constant *, 8> ReturnCSVs;

  model::TypePath PrototypePath = getPrototype(Binary,
                                               Entry,
                                               CallSiteBlock.ID(),
                                               CallSite);
  auto Prototype = abi::FunctionType::Layout::make(PrototypePath);
  revng_assert(Prototype.verify());

  bool IsIndirect = (Callee.getCallee() == FunctionDispatcher);
  if (IsIndirect) {
    // Create a new `indirect_placeholder` function with the specific function
    // type we need
    Value *PC = GCBI.programCounterHandler()->loadJumpablePC(Builder);
    auto [ReturnType, Arguments] = getLLVMReturnTypeAndArguments(&M, Prototype);
    auto *NewType = FunctionType::get(ReturnType, Arguments, false);
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
  for (model::Register::Values Register : Prototype.argumentRegisters())
    Arguments.push_back(loadCSVOrUndef(Builder, &M, Register));

  for (model::Register::Values Register : Prototype.returnValueRegisters())
    ReturnCSVs.push_back(getCSVOrUndef(&M, Register).second);

  //
  // Produce the call
  //
  auto *Result = Builder.CreateCall(Callee, Arguments);
  Result->addFnAttr(Attribute::NoMerge);
  revng_assert(Result->getDebugLoc());

  if (ReturnCSVs.size() != 1) {
    unsigned I = 0;
    for (Constant *ReturnCSV : ReturnCSVs) {
      Builder.CreateStore(Builder.CreateExtractValue(Result, { I }), ReturnCSV);
      I++;
    }
  } else {
    Builder.CreateStore(Result, ReturnCSVs[0]);
  }

  return Result;
}

void EnforceABI::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<FunctionMetadataCachePass>();
  AU.setPreservesAll();
}

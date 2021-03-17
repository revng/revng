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
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/ADT/LazySmallBitVector.h"
#include "revng/ADT/SmallMap.h"
#include "revng/FunctionIsolation/EnforceABI.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

using namespace llvm;

using StackAnalysis::FunctionCallRegisterArgument;
using StackAnalysis::FunctionCallReturnValue;
using StackAnalysis::FunctionRegisterArgument;
using StackAnalysis::FunctionReturnValue;
using StackAnalysis::FunctionsSummary;

using CallSiteDescription = FunctionsSummary::CallSiteDescription;
using FunctionDescription = FunctionsSummary::FunctionDescription;
using FCRD = FunctionsSummary::FunctionCallRegisterDescription;
using FunctionCallRegisterDescription = FCRD;
using FRD = FunctionsSummary::FunctionRegisterDescription;
using FunctionRegisterDescription = FRD;

char EnforceABI::ID = 0;
using Register = RegisterPass<EnforceABI>;
static Register X("enforce-abi", "Enforce ABI Pass", true, true);

static Logger<> EnforceABILog("enforce-abi");
static cl::opt<bool> DisableSafetyChecks("disable-enforce-abi-safety-checks",
                                         cl::desc("Disable safety checks in "
                                                  " ABI enforcing"),
                                         cl::cat(MainCategory),
                                         cl::init(false));

static bool shouldEmit(model::RegisterState::Values V) {
  return (V == model::RegisterState::Yes or V == model::RegisterState::YesOrDead
          or V == model::RegisterState::Dead);
}

static bool areCompatible(model::RegisterState::Values LHS,
                          model::RegisterState::Values RHS) {
  using namespace model::RegisterState;

  if (LHS == RHS or LHS == Maybe or RHS == Maybe)
    return true;

  switch (LHS) {
  case NoOrDead:
    return RHS == No or RHS == Dead;
  case YesOrDead:
    return RHS == Yes or RHS == Dead;
  case No:
    return RHS == NoOrDead;
  case Yes:
    return RHS == YesOrDead;
  case Dead:
    return RHS == NoOrDead or RHS == YesOrDead;
  case Contradiction:
    return false;
  case Invalid:
  default:
    revng_abort();
  }

  revng_abort();
}

static bool areCompatible(const model::FunctionABIRegister &LHS,
                          const model::FunctionABIRegister &RHS) {
  return areCompatible(LHS.Argument, RHS.Argument)
         and areCompatible(LHS.ReturnValue, RHS.ReturnValue);
}

static StringRef areCompatible(const model::Function &Callee,
                               const model::FunctionEdge &CallSite) {
  for (const model::FunctionABIRegister &Register : Callee.Registers) {
    auto It = CallSite.Registers.find(Register.Register);
    if (It != CallSite.Registers.end() and not areCompatible(Register, *It)) {
      return model::Register::getName(Register.Register);
    }
  }

  return StringRef();
}

class HelperCallSite {
public:
  using CSVToIndexMap = std::map<GlobalVariable *, unsigned>;
  HelperCallSite(const std::vector<GlobalVariable *> &ReadCSVs,
                 const std::vector<GlobalVariable *> &WrittenCSVs,
                 const CSVToIndexMap &CSVToIndex,
                 Function *Helper) :
    Helper(Helper) {
    revng_assert(Helper != nullptr);

    populateBitmap(CSVToIndex, ReadCSVs, Read);
    populateBitmap(CSVToIndex, WrittenCSVs, Written);
  }

public:
  bool operator<(const HelperCallSite &Other) const {
    auto ThisTie = std::tie(Helper, Read, Written);
    auto OtherTie = std::tie(Other.Helper, Other.Read, Other.Written);
    return ThisTie < OtherTie;
  }

  bool operator==(const HelperCallSite &Other) const {
    auto ThisTie = std::tie(Helper, Read, Written);
    auto OtherTie = std::tie(Other.Helper, Other.Read, Other.Written);
    return ThisTie == OtherTie;
  }

  Function *helper() const { return Helper; }

private:
  static void populateBitmap(const CSVToIndexMap &CSVToIndex,
                             const std::vector<GlobalVariable *> &Source,
                             LazySmallBitVector &Target) {
    for (GlobalVariable *CSV : Source) {
      Target.set(CSVToIndex.at(CSV));
    }
  }

private:
  Function *Helper;
  LazySmallBitVector Read;
  LazySmallBitVector Written;
};

class EnforceABIImpl {
public:
  EnforceABIImpl(Module &M,
                 GeneratedCodeBasicInfo &GCBI,
                 const model::Binary &Binary) :
    M(M),
    GCBI(GCBI),
    FunctionDispatcher(M.getFunction("function_dispatcher")),
    Context(M.getContext()),
    IndirectPlaceholderPool(&M, false),
    Binary(Binary) {}

  void run();

private:
  Function *handleFunction(Function &F, const model::Function &FunctionModel);

  void handleRegularFunctionCall(CallInst *Call);
  void handleHelperFunctionCall(CallInst *Call);
  void generateCall(IRBuilder<> &Builder,
                    Function *Callee,
                    const model::FunctionEdge &CallSite);
  void replaceCSVsWithAlloca();
  void handleRoot();

private:
  Module &M;
  GeneratedCodeBasicInfo &GCBI;
  std::map<Function *, const model::Function *> FunctionsMap;
  std::map<Function *, Function *> OldToNew;
  Function *FunctionDispatcher;
  Function *OpaquePC;
  LLVMContext &Context;
  std::map<HelperCallSite, Function *> HelperCallSites;
  std::map<GlobalVariable *, unsigned> CSVToIndex;
  OpaqueFunctionsPool<FunctionType *> IndirectPlaceholderPool;
  const model::Binary &Binary;
};

bool EnforceABI::runOnModule(Module &M) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  const auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Binary = ModelWrapper.getReadOnlyModel();

  EnforceABIImpl Impl(M, GCBI, Binary);
  Impl.run();
  return false;
}

// TODO: assign alias information
static Function *
createHelperWrapper(Function *Helper,
                    const std::vector<GlobalVariable *> &Read,
                    const std::vector<GlobalVariable *> &Written) {
  auto *PointeeTy = Helper->getType()->getPointerElementType();
  auto *HelperType = cast<FunctionType>(PointeeTy);

  //
  // Create new argument list
  //
  SmallVector<Type *, 16> NewArguments;

  // Initialize with base arguments
  std::copy(HelperType->param_begin(),
            HelperType->param_end(),
            std::back_inserter(NewArguments));

  // Add type of read registers
  for (GlobalVariable *CSV : Read)
    NewArguments.push_back(CSV->getType()->getPointerElementType());

  //
  // Create return type
  //

  // If the helpers does not write any register, reuse the original
  // return type
  Type *OriginalReturnType = HelperType->getReturnType();
  Type *NewReturnType = OriginalReturnType;

  bool HasOutputCSVs = Written.size() != 0;
  bool OriginalWasVoid = OriginalReturnType->isVoidTy();
  if (HasOutputCSVs) {
    SmallVector<Type *, 16> ReturnTypes;

    // If the original return type was not void, put it as first field
    // in the return type struct
    if (not OriginalWasVoid) {
      ReturnTypes.push_back(OriginalReturnType);
    }

    for (GlobalVariable *CSV : Written)
      ReturnTypes.push_back(CSV->getType()->getPointerElementType());

    NewReturnType = StructType::create(ReturnTypes);
  }

  //
  // Create new helper wrapper function
  //
  auto *NewHelperType = FunctionType::get(NewReturnType, NewArguments, false);
  auto *HelperWrapper = Function::Create(NewHelperType,
                                         Helper->getLinkage(),
                                         Twine(Helper->getName()) + "_wrapper",
                                         Helper->getParent());

  auto *Entry = BasicBlock::Create(getContext(Helper), "", HelperWrapper);

  //
  // Populate the helper wrapper function
  //
  IRBuilder<> Builder(Entry);

  // Serialize read CSV
  auto It = HelperWrapper->arg_begin();
  for (unsigned I = 0; I < HelperType->getNumParams(); I++, It++) {
    // Do nothing
    revng_assert(It != HelperWrapper->arg_end());
  }

  for (GlobalVariable *CSV : Read) {
    revng_assert(It != HelperWrapper->arg_end());
    Builder.CreateStore(&*It, CSV);
    It++;
  }
  revng_assert(It == HelperWrapper->arg_end());

  // Prepare the arguments
  SmallVector<Value *, 16> HelperArguments;
  It = HelperWrapper->arg_begin();
  for (unsigned I = 0; I < HelperType->getNumParams(); I++, It++) {
    revng_assert(It != HelperWrapper->arg_end());
    HelperArguments.push_back(&*It);
  }

  // Create the function call
  auto *HelperResult = Builder.CreateCall(Helper, HelperArguments);

  // Deserialize and return the appropriate values
  if (HasOutputCSVs) {
    SmallVector<Value *, 16> ReturnValues;

    if (not OriginalWasVoid)
      ReturnValues.push_back(HelperResult);

    for (GlobalVariable *CSV : Written)
      ReturnValues.push_back(Builder.CreateLoad(CSV));

    Builder.CreateAggregateRet(ReturnValues.data(), ReturnValues.size());

  } else if (OriginalWasVoid) {
    Builder.CreateRetVoid();
  } else {
    Builder.CreateRet(HelperResult);
  }

  return HelperWrapper;
}

void EnforceABIImpl::run() {
  // Assign an index to each CSV
  unsigned I = 0;
  for (GlobalVariable *CSV : GCBI.csvs()) {
    CSVToIndex[CSV] = I;
    I++;
  }

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
  OpaquePC->addFnAttr(Attribute::ReadOnly);

  std::vector<Function *> OldFunctions;
  for (const model::Function &FunctionModel : Binary.Functions) {
    if (FunctionModel.Type == model::FunctionType::Fake)
      continue;

    revng_assert(FunctionModel.Name.size() != 0);
    Function *OldFunction = M.getFunction(FunctionModel.Name);
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

  // Handle function calls to helpers in isolated functions
  std::vector<CallInst *> HelperCalls;
  for (Function &F : M.functions())
    if (isHelper(&F))
      for (User *U : F.users())
        if (auto *Call = dyn_cast<CallInst>(skipCasts(U)))
          if (IsInIsolatedFunction(Call))
            HelperCalls.push_back(Call);

  for (CallInst *Call : HelperCalls)
    handleHelperFunctionCall(Call);

  // Handle invoke instructions in `root`
  handleRoot();

  // Promote CSVs to allocas
  replaceCSVsWithAlloca();

  // Drop function_dispatcher
  if (FunctionDispatcher != nullptr) {
    FunctionDispatcher->deleteBody();
    ReturnInst::Create(Context,
                       BasicBlock::Create(Context, "", FunctionDispatcher));
  }

  // Drop all the old functions, after we stole all of its blocks
  for (Function *OldFunction : OldFunctions) {
    for (User *U : OldFunction->users())
      cast<Instruction>(U)->getParent()->dump();
    OldFunction->eraseFromParent();
  }

  // Quick and dirty DCE
  for (auto &P : FunctionsMap) {
    Function &F = *P.first;

    std::set<BasicBlock *> Reachable;
    ReversePostOrderTraversal<BasicBlock *> RPOT(&F.getEntryBlock());
    for (BasicBlock *BB : RPOT)
      Reachable.insert(BB);

    std::vector<BasicBlock *> ToDelete;
    for (BasicBlock &BB : F)
      if (Reachable.count(&BB) == 0)
        ToDelete.push_back(&BB);

    for (BasicBlock *BB : ToDelete)
      BB->dropAllReferences();

    for (BasicBlock *BB : ToDelete)
      BB->eraseFromParent();
  }

  if (VerifyLog.isEnabled()) {
    raw_os_ostream Stream(dbg);
    revng_assert(not verifyModule(M, &Stream));
  }
}

void EnforceABIImpl::handleRoot() {
  // Handle invokes in root
  Function *Root = M.getFunction("root");
  revng_assert(Root != nullptr);

  for (BasicBlock &BB : *Root) {
    // Find invoke instruction
    auto *Invoke = dyn_cast<InvokeInst>(BB.getTerminator());

    if (Invoke == nullptr)
      continue;

    revng_assert(BB.size() == 1);

    Function *Callee = OldToNew.at(Invoke->getCalledFunction());
    const model::Function *Function = FunctionsMap.at(Callee);

    // Collect arguments
    IRBuilder<> Builder(Invoke);
    std::vector<Value *> Arguments;
    for (const model::FunctionABIRegister &Register : Function->Registers) {
      if (shouldEmit(Register.Argument)) {
        auto Name = ABIRegister::toCSVName(Register.Register);
        GlobalVariable *CSV = M.getGlobalVariable(Name, true);
        revng_assert(CSV != nullptr);
        Arguments.push_back(Builder.CreateLoad(CSV));
      }
    }

    // Create the new invoke with the appropriate arguments
    auto *NewInvoke = Builder.CreateInvoke(Callee,
                                           Invoke->getNormalDest(),
                                           Invoke->getUnwindDest(),
                                           Arguments);

    // Erase the old invoke
    Invoke->eraseFromParent();

    // TODO: handle return values
  }
}

Function *EnforceABIImpl::handleFunction(Function &OldFunction,
                                         const model::Function &FunctionModel) {
  SmallVector<Type *, 8> ArgumentsTypes;
  SmallVector<GlobalVariable *, 8> ArgumentCSVs;
  SmallVector<Type *, 8> ReturnTypes;
  SmallVector<GlobalVariable *, 8> ReturnCSVs;

  for (const model::FunctionABIRegister &Register : FunctionModel.Registers) {
    auto Name = ABIRegister::toCSVName(Register.Register);
    auto *CSV = cast<GlobalVariable>(M.getGlobalVariable(Name, true));

    // Collect arguments
    if (shouldEmit(Register.Argument)) {
      ArgumentsTypes.push_back(CSV->getType()->getPointerElementType());
      ArgumentCSVs.push_back(CSV);
    }

    // Collect return values
    if (shouldEmit(Register.ReturnValue)) {
      ReturnTypes.push_back(CSV->getType()->getPointerElementType());
      ReturnCSVs.push_back(CSV);
    }
  }

  // Create the return type
  Type *ReturnType = Type::getVoidTy(Context);
  if (ReturnTypes.size() == 0)
    ReturnType = Type::getVoidTy(Context);
  else if (ReturnTypes.size() == 1)
    ReturnType = ReturnTypes[0];
  else
    ReturnType = StructType::create(ReturnTypes);

  // Create new function
  auto *NewType = FunctionType::get(ReturnType, ArgumentsTypes, false);
  auto *NewFunction = Function::Create(NewType,
                                       GlobalValue::ExternalLinkage,
                                       "",
                                       OldFunction.getParent());
  NewFunction->takeName(&OldFunction);
  NewFunction->copyAttributesFrom(&OldFunction);

  // Set argument names
  unsigned I = 0;
  for (Argument &Argument : NewFunction->args())
    Argument.setName(ArgumentCSVs[I++]->getName());

  // Steal body from the old function
  std::vector<BasicBlock *> Body;
  for (BasicBlock &BB : OldFunction)
    Body.push_back(&BB);
  auto &NewBody = NewFunction->getBasicBlockList();
  for (BasicBlock *BB : Body) {
    BB->removeFromParent();
    revng_assert(BB->getParent() == nullptr);
    NewBody.push_back(BB);
    revng_assert(BB->getParent() == NewFunction);
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

        if (ReturnTypes.size() == 1)
          Builder.CreateRet(ReturnValues[0]);
        else
          Builder.CreateAggregateRet(ReturnValues.data(), ReturnValues.size());

        Return->eraseFromParent();
      }
    }
  }

  return NewFunction;
}

void EnforceABIImpl::handleHelperFunctionCall(CallInst *Call) {
  using namespace StackAnalysis;
  Function *Helper = cast<Function>(skipCasts(Call->getCalledValue()));

  auto UsedCSVs = GeneratedCodeBasicInfo::getCSVUsedByHelperCall(Call);
  UsedCSVs.sort();

  auto *PointeeTy = Helper->getType()->getPointerElementType();
  auto *HelperType = cast<llvm::FunctionType>(PointeeTy);

  HelperCallSite CallSite(UsedCSVs.Read, UsedCSVs.Written, CSVToIndex, Helper);
  auto It = HelperCallSites.find(CallSite);

  Function *HelperWrapper = nullptr;
  if (It != HelperCallSites.end()) {
    revng_assert(CallSite == It->first);
    HelperWrapper = It->second;
  } else {
    HelperWrapper = createHelperWrapper(Helper,
                                        UsedCSVs.Read,
                                        UsedCSVs.Written);

    // Record the new wrapper for future use
    HelperCallSites[CallSite] = HelperWrapper;
  }

  //
  // Emit call to the helper wrapper
  //
  IRBuilder<> Builder(Call);

  // Initialize the new set of arguments with the old ones
  SmallVector<Value *, 16> NewArguments;
  for (auto [Argument, Type] : zip(Call->args(), HelperType->params()))
    NewArguments.push_back(Builder.CreateBitOrPointerCast(Argument, Type));

  // Add arguments read
  for (GlobalVariable *CSV : UsedCSVs.Read)
    NewArguments.push_back(Builder.CreateLoad(CSV));

  // Emit the actual call
  Value *Result = Builder.CreateCall(HelperWrapper, NewArguments);

  bool HasOutputCSVs = UsedCSVs.Written.size() != 0;
  bool OriginalWasVoid = HelperType->getReturnType()->isVoidTy();
  if (HasOutputCSVs) {

    unsigned FirstDeserialized = 0;
    if (not OriginalWasVoid) {
      FirstDeserialized = 1;
      // RAUW the new result
      Value *HelperResult = Builder.CreateExtractValue(Result, { 0 });
      Call->replaceAllUsesWith(HelperResult);
    }

    // Restore into CSV the written registers
    for (unsigned I = 0; I < UsedCSVs.Written.size(); I++) {
      unsigned ResultIndex = { FirstDeserialized + I };
      Builder.CreateStore(Builder.CreateExtractValue(Result, ResultIndex),
                          UsedCSVs.Written[I]);
    }

  } else if (not OriginalWasVoid) {
    Call->replaceAllUsesWith(Result);
  }

  // Erase the old call
  Call->eraseFromParent();
}

void EnforceABIImpl::handleRegularFunctionCall(CallInst *Call) {
  Function *Caller = Call->getParent()->getParent();
  const model::Function &FunctionModel = *FunctionsMap.at(Caller);

  revng_assert(Call->getParent()->getParent()->getName() == FunctionModel.Name);

  Function *Callee = cast<Function>(skipCasts(Call->getCalledValue()));
  bool IsDirect = (Callee != FunctionDispatcher);
  if (IsDirect)
    Callee = OldToNew.at(Callee);

  // Identify the corresponding call site in the model
  MetaAddress BasicBlockAddress = GCBI.getJumpTarget(Call->getParent());
  const model::BasicBlock &Block = FunctionModel.CFG.at(BasicBlockAddress);
  const model::FunctionEdge *CallSite = nullptr;
  for (const model::FunctionEdge &Edge : Block.Successors) {
    using namespace model::FunctionEdgeType;
    if (Edge.Type == FunctionCall or Edge.Type == IndirectCall
        or Edge.Type == IndirectTailCall) {
      CallSite = &Edge;
      break;
    }
  }

  if (DisableSafetyChecks or IsDirect) {
    // The callee is a well-known callee, generate a direct call
    IRBuilder<> Builder(Call);
    generateCall(Builder, Callee, *CallSite);

    // Create an additional store to the local %pc, so that the optimizer cannot
    // do stuff with llvm.assume.
    revng_assert(OpaquePC != nullptr);
    Builder.CreateStore(Builder.CreateCall(OpaquePC), GCBI.pcReg());

  } else {
    // If it's an indirect call, enumerate all the compatible callees and
    // generate a call for each of them

    EnforceABILog << getName(Call) << " is an indirect call compatible with:\n";

    BasicBlock *BeforeSplit = Call->getParent();
    BasicBlock *AfterSplit = BeforeSplit->splitBasicBlock(Call);
    BeforeSplit->getTerminator()->eraseFromParent();

    IRBuilder<> Builder(BeforeSplit);
    BasicBlock *UnexpectedPC = findByBlockType(AfterSplit->getParent(),
                                               BlockType::UnexpectedPCBlock);

    ProgramCounterHandler::DispatcherTargets Targets;

    unsigned Count = 0;
    for (auto &[F, FunctionModel] : FunctionsMap) {
      EnforceABILog << "  " << F->getName().data() << " ";

      // Check compatibility
      StringRef IncompatibleCSV = areCompatible(*FunctionModel, *CallSite);
      bool Incompatible = not IncompatibleCSV.empty();
      if (Incompatible) {
        EnforceABILog << "[No: " << IncompatibleCSV.data() << "]";
      } else {
        EnforceABILog << "[Yes]";
        Count++;

        // Create the basic block containing the call
        auto *Case = BasicBlock::Create(Context,
                                        "",
                                        BeforeSplit->getParent(),
                                        AfterSplit);
        Builder.SetInsertPoint(Case);
        generateCall(Builder, F, *CallSite);
        Builder.CreateBr(AfterSplit);

        // Record for inline dispatcher
        Targets.push_back({ FunctionModel->Entry, Case });
      }
      EnforceABILog << DoLog;
    }

    // Actually create the inline dispatcher
    Builder.SetInsertPoint(BeforeSplit);
    GCBI.programCounterHandler()->buildDispatcher(Targets,
                                                  Builder,
                                                  UnexpectedPC,
                                                  {});

    EnforceABILog << Count << " functions" << DoLog;
  }

  // Drop the original call
  Call->eraseFromParent();
}

void EnforceABIImpl::generateCall(IRBuilder<> &Builder,
                                  Function *Callee,
                                  const model::FunctionEdge &CallSite) {
  revng_assert(Callee != nullptr);

  llvm::SmallVector<Type *, 8> ArgumentsTypes;
  llvm::SmallVector<Value *, 8> Arguments;
  llvm::SmallVector<Type *, 8> ReturnTypes;
  llvm::SmallVector<GlobalVariable *, 8> ReturnCSVs;

  bool IsDirect = (Callee != FunctionDispatcher);
  if (not IsDirect) {
    revng_assert(DisableSafetyChecks);

    // Collect arguments, returns and their type.
    for (const model::FunctionABIRegister &Register : CallSite.Registers) {
      auto Name = ABIRegister::toCSVName(Register.Register);
      GlobalVariable *CSV = M.getGlobalVariable(Name, true);
      if (shouldEmit(Register.Argument)) {
        ArgumentsTypes.push_back(CSV->getType()->getPointerElementType());
        Arguments.push_back(Builder.CreateLoad(CSV));
      }

      if (shouldEmit(Register.ReturnValue)) {
        ReturnTypes.push_back(CSV->getType()->getPointerElementType());
        ReturnCSVs.push_back(CSV);
      }
    }

    // Create here on the fly the indirect function that we want to call.
    // Create the return type
    Type *ReturnType = Type::getVoidTy(Context);
    if (ReturnTypes.size() == 0)
      ReturnType = Type::getVoidTy(Context);
    else if (ReturnTypes.size() == 1)
      ReturnType = ReturnTypes[0];
    else
      ReturnType = StructType::create(ReturnTypes);

    // Create a new `indirect_placeholder` function with the specific function
    // type we need
    auto *NewType = FunctionType::get(ReturnType, ArgumentsTypes, false);
    Callee = IndirectPlaceholderPool.get(NewType,
                                         NewType,
                                         "indirect_placeholder");
  } else {

    // Additional debug checks if we are not emitting an indirect call.
    BasicBlock *InsertBlock = Builder.GetInsertPoint()->getParent();
    revng_log(EnforceABILog,
              "Emitting call to " << getName(Callee) << " from "
                                  << getName(InsertBlock));

    const model::Function *FunctionModel = FunctionsMap.at(Callee);
    revng_assert(Callee->getName().startswith("bb."));
    StringRef IncompatibleCSV = areCompatible(*FunctionModel, CallSite);
    bool Incompatible = not IncompatibleCSV.empty();
    if (Incompatible) {
      dbg << getName(InsertBlock) << " -> "
          << (Callee == nullptr ? "nullptr" : Callee->getName().data()) << ": "
          << IncompatibleCSV.data() << "\n";
      revng_abort();
    }

    // Collect arguments, returns and their type.
    for (const model::FunctionABIRegister &Register :
         FunctionModel->Registers) {
      auto Name = ABIRegister::toCSVName(Register.Register);
      GlobalVariable *CSV = M.getGlobalVariable(Name, true);

      if (shouldEmit(Register.Argument)) {
        ArgumentsTypes.push_back(CSV->getType()->getPointerElementType());
        Arguments.push_back(Builder.CreateLoad(CSV));
      }

      if (shouldEmit(Register.ReturnValue)) {
        ReturnTypes.push_back(CSV->getType()->getPointerElementType());
        ReturnCSVs.push_back(CSV);
      }
    }
  }

  auto *Result = Builder.CreateCall(Callee, Arguments);
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

void EnforceABIImpl::replaceCSVsWithAlloca() {
  OpaqueFunctionsPool<StringRef> CSVInitializers(&M, false);
  CSVInitializers.addFnAttribute(Attribute::ReadOnly);
  CSVInitializers.addFnAttribute(Attribute::NoUnwind);

  // Each CSV has a map. The key of the map is a Funciton, and the mapped value
  // is an AllocaInst used to locally represent that CSV.
  using MapsVector = std::vector<ValueMap<Function *, Value *>>;
  MapsVector CSVMaps(GCBI.csvs().size(), {});

  // Map CSV GlobalVariable * to a position in CSVMaps
  ValueMap<llvm::GlobalVariable *, MapsVector::size_type> CSVPosition;
  for (auto &Group : llvm::enumerate(GCBI.csvs()))
    CSVPosition[Group.value()] = Group.index();

  for (auto &P : FunctionsMap) {
    Function &F = *P.first;

    // Identifies the GlobalVariables representing CSVs used in F.
    std::set<GlobalVariable *> CSVs;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        Value *Pointer = nullptr;
        if (auto *Load = dyn_cast<LoadInst>(&I))
          Pointer = Load->getPointerOperand();
        else if (auto *Store = dyn_cast<StoreInst>(&I))
          Pointer = Store->getPointerOperand();
        else
          continue;

        Pointer = skipCasts(Pointer);

        if (auto *CSV = dyn_cast_or_null<GlobalVariable>(Pointer))
          if (not GCBI.isSPReg(CSV))
            CSVs.insert(CSV);
      }
    }

    // Create an alloca for each CSV and replace all uses of CSVs with the
    // corresponding allocas
    BasicBlock &Entry = F.getEntryBlock();
    IRBuilder<> InitializersBuilder(&Entry, Entry.begin());
    auto *Separator = InitializersBuilder.CreateUnreachable();
    IRBuilder<> AllocaBuilder(Separator);

    // For each GlobalVariable representing a CSV used in F, create a dedicated
    // alloca and save it in CSVMaps.
    for (GlobalVariable *CSV : CSVs) {

      Type *CSVType = CSV->getType()->getPointerElementType();

      // Create and register the alloca
      auto *Alloca = AllocaBuilder.CreateAlloca(CSVType,
                                                nullptr,
                                                CSV->getName());

      // Initialize all allocas with opaque, CSV-specific values
      auto *Initializer = CSVInitializers.get(CSV->getName(),
                                              CSVType,
                                              {},
                                              Twine("init_") + CSV->getName());
      {
        auto &B = InitializersBuilder;
        B.CreateStore(B.CreateCall(Initializer), Alloca);
      }

      // Ignore it if it's not a CSV
      auto CSVPosIt = CSVPosition.find(CSV);
      if (CSVPosIt != CSVPosition.end()) {
        auto CSVPos = CSVPosIt->second;
        CSVMaps[CSVPos][&F] = Alloca;
      }
    }

    Separator->eraseFromParent();
  }

  // Substitute the uses of the GlobalVariables representing the CSVs with the
  // dedicate AllocaInst that were created in each Function.
  for (GlobalVariable *CSV : GCBI.csvs()) {
    auto CSVPosIt = CSVPosition.find(CSV);
    auto CSVPos = CSVPosIt->second;
    const auto &FunctionAllocas = CSVMaps.at(CSVPos);
    if (not FunctionAllocas.empty())
      replaceAllUsesInFunctionsWith(CSV, FunctionAllocas);
  }
}

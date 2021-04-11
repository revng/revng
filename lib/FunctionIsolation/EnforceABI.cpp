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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/ADT/LazySmallBitVector.h"
#include "revng/ADT/SmallMap.h"
#include "revng/FunctionIsolation/EnforceABI.h"
#include "revng/FunctionIsolation/StructInitializers.h"
#include "revng/Support/FunctionTags.h"
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

class EnforceABIImpl {
public:
  EnforceABIImpl(Module &M,
                 GeneratedCodeBasicInfo &GCBI,
                 const model::Binary &Binary) :
    M(M),
    GCBI(GCBI),
    FunctionDispatcher(M.getFunction("function_dispatcher")),
    Context(M.getContext()),
    Initializers(&M),
    IndirectPlaceholderPool(&M, false),
    Binary(Binary) {}

  void run();

private:
  Function *handleFunction(Function &F, const model::Function &FunctionModel);

  void handleRegularFunctionCall(CallInst *Call);
  void generateCall(IRBuilder<> &Builder,
                    Function *Callee,
                    const model::FunctionEdge &CallSite);
  void handleRoot();

private:
  Module &M;
  GeneratedCodeBasicInfo &GCBI;
  std::map<Function *, const model::Function *> FunctionsMap;
  std::map<Function *, Function *> OldToNew;
  Function *FunctionDispatcher;
  Function *OpaquePC;
  LLVMContext &Context;
  StructInitializers Initializers;
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
  OpaquePC->addFnAttr(Attribute::ReadOnly);
  FunctionTags::OpaqueCSVValue.addTo(OpaquePC);

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
  for (auto [F, _] : FunctionsMap)
    EliminateUnreachableBlocks(*F, nullptr, false);

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
  FunctionTags::Lifted.addTo(NewFunction);

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
          Initializers.createReturn(Builder, ReturnValues);

        Return->eraseFromParent();
      }
    }
  }

  return NewFunction;
}

void EnforceABIImpl::handleRegularFunctionCall(CallInst *Call) {
  Function *Caller = Call->getParent()->getParent();
  const model::Function &FunctionModel = *FunctionsMap.at(Caller);

  revng_assert(Call->getParent()->getParent()->getName() == FunctionModel.Name);

  Function *Callee = cast<Function>(skipCasts(Call->getCalledOperand()));
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
    revng_assert(FunctionTags::Lifted.isTagOf(Callee));
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

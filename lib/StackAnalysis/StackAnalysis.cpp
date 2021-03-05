/// \file StackAnalysis.cpp
/// \brief Implementation of the stack analysis, which provides information
///        about function boundaries, basic block types, arguments and return
///        values.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "revng/ADT/Queue.h"
#include "revng/ABIAnalyses/ABIAnalysis.h"
#include "revng/BasicAnalyses/CSVAliasAnalysis.h"
#include "revng/BasicAnalyses/PromoteGlobalToLocalVars.h"
#include "revng/BasicAnalyses/RemoveHelperCalls.h"
#include "revng/BasicAnalyses/RemoveNewPCCalls.h"
#include "revng/Model/Binary.h"
#include "revng/StackAnalysis/StackAnalysis.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/IRHelpers.h"

#include "Cache.h"
#include "InterproceduralAnalysis.h"
#include "Intraprocedural.h"

using llvm::BasicBlock;
using llvm::Function;
using llvm::Module;
using llvm::RegisterPass;

using FunctionEdgeTypeValue = model::FunctionEdgeType::Values;
using FunctionTypeValue = model::FunctionType::Values;
using GCBI = GeneratedCodeBasicInfo;

static Logger<> ClobberedLog("clobbered");
static Logger<> StackAnalysisLog("stackanalysis");
static Logger<> CFEPLog("cfep");

using namespace llvm::cl;

namespace StackAnalysis {

const std::set<llvm::GlobalVariable *> EmptyCSVSet;

char StackAnalysis::ID = 0;

using RegisterABI = RegisterPass<StackAnalysis>;
static RegisterABI Y("abi-analysis", "ABI Analysis Pass", true, true);

static opt<std::string> ABIAnalysisOutputPath("abi-analysis-output",
                                              desc("Destination path for the "
                                                   "ABI Analysis Pass"),
                                              value_desc("path"),
                                              cat(MainCategory));

template<bool FunctionCall>
static model::RegisterState::Values
toRegisterState(RegisterArgument<FunctionCall> RA) {
  switch (RA.value()) {
  case RegisterArgument<FunctionCall>::NoOrDead:
    return model::RegisterState::NoOrDead;
  case RegisterArgument<FunctionCall>::Maybe:
    return model::RegisterState::Maybe;
  case RegisterArgument<FunctionCall>::Yes:
    return model::RegisterState::Yes;
  case RegisterArgument<FunctionCall>::Dead:
    return model::RegisterState::Dead;
  case RegisterArgument<FunctionCall>::Contradiction:
    return model::RegisterState::Contradiction;
  case RegisterArgument<FunctionCall>::No:
    return model::RegisterState::No;
  }

  revng_abort();
}

static model::RegisterState::Values toRegisterState(FunctionReturnValue RV) {
  switch (RV.value()) {
  case FunctionReturnValue::No:
    return model::RegisterState::No;
  case FunctionReturnValue::NoOrDead:
    return model::RegisterState::NoOrDead;
  case FunctionReturnValue::YesOrDead:
    return model::RegisterState::YesOrDead;
  case FunctionReturnValue::Maybe:
    return model::RegisterState::Maybe;
  case FunctionReturnValue::Contradiction:
    return model::RegisterState::Contradiction;
  }

  revng_abort();
}

static model::RegisterState::Values
toRegisterState(FunctionCallReturnValue RV) {
  switch (RV.value()) {
  case FunctionCallReturnValue::No:
    return model::RegisterState::No;
  case FunctionCallReturnValue::NoOrDead:
    return model::RegisterState::NoOrDead;
  case FunctionCallReturnValue::YesOrDead:
    return model::RegisterState::YesOrDead;
  case FunctionCallReturnValue::Yes:
    return model::RegisterState::Yes;
  case FunctionCallReturnValue::Dead:
    return model::RegisterState::Dead;
  case FunctionCallReturnValue::Maybe:
    return model::RegisterState::Maybe;
  case FunctionCallReturnValue::Contradiction:
    return model::RegisterState::Contradiction;
  }

  revng_abort();
}

void commitToModel(GeneratedCodeBasicInfo &GCBI,
                   Function *F,
                   const FunctionsSummary &Summary,
                   model::Binary &TheBinary);

void commitToModel(GeneratedCodeBasicInfo &GCBI,
                   Function *F,
                   const FunctionsSummary &Summary,
                   model::Binary &TheBinary) {
  using namespace model;

  for (const auto &[Entry, FunctionSummary] : Summary.Functions) {
    if (Entry == nullptr)
      continue;

    //
    // Initialize model::Function
    //

    // Get the entry point address
    MetaAddress EntryPC = getBasicBlockPC(Entry);
    revng_assert(EntryPC.isValid());

    // Create the function
    revng_assert(TheBinary.Functions.count(EntryPC) == 0);
    model::Function &Function = TheBinary.Functions[EntryPC];

    // Assign a name
    Function.Name = Entry->getName();
    revng_assert(Function.Name.size() != 0);

    using FT = model::FunctionType::Values;
    Function.Type = static_cast<FT>(FunctionSummary.Type);

    // Populate arguments and return values
    {
      auto Inserter = Function.Registers.batch_insert();
      for (auto &[CSV, FRD] : FunctionSummary.RegisterSlots) {
        auto ID = ABIRegister::fromCSVName(CSV->getName(), GCBI.arch());
        if (ID == model::Register::Invalid)
          continue;
        FunctionABIRegister TheRegister(ID);
        TheRegister.Argument = toRegisterState(FRD.Argument);
        TheRegister.ReturnValue = toRegisterState(FRD.ReturnValue);
        Inserter.insert(TheRegister);
      }
    }

    for (auto &[BB, Branch] : FunctionSummary.BasicBlocks) {
      // Remap BranchType to FunctionEdgeType
      namespace FET = FunctionEdgeType;
      FET::Values EdgeType = FET::Invalid;
      bool IsCall = false;

      switch (Branch) {
      case BranchType::Invalid:
      case BranchType::FakeFunction:
      case BranchType::RegularFunction:
      case BranchType::NoReturnFunction:
      case BranchType::UnhandledCall:
        revng_abort();
        break;

      case BranchType::InstructionLocalCFG:
        EdgeType = FET::Invalid;
        break;

      case BranchType::FunctionLocalCFG:
        EdgeType = FET::DirectBranch;
        break;

      case BranchType::FakeFunctionCall:
        EdgeType = FET::FakeFunctionCall;
        break;

      case BranchType::FakeFunctionReturn:
        EdgeType = FET::FakeFunctionReturn;
        break;

      case BranchType::HandledCall:
        IsCall = true;
        EdgeType = FET::FunctionCall;
        break;

      case BranchType::IndirectCall:
        IsCall = true;
        EdgeType = FET::IndirectCall;
        break;

      case BranchType::Return:
        EdgeType = FET::Return;
        break;

      case BranchType::BrokenReturn:
        EdgeType = FET::BrokenReturn;
        break;

      case BranchType::IndirectTailCall:
        IsCall = true;
        EdgeType = FET::IndirectTailCall;
        break;

      case BranchType::LongJmp:
        EdgeType = FET::LongJmp;
        break;

      case BranchType::Killer:
        EdgeType = FET::Killer;
        break;

      case BranchType::Unreachable:
        EdgeType = FET::Unreachable;
        break;
      }

      if (EdgeType == FET::Invalid)
        continue;

      // Identify Source address
      auto [Source, Size] = getPC(BB->getTerminator());
      Source += Size;
      revng_assert(Source.isValid());

      // Identify Destination address
      llvm::BasicBlock *JumpTargetBB = GCBI.getJumpTargetBlock(BB);
      MetaAddress JumpTargetAddress = GCBI.getPCFromNewPC(JumpTargetBB);
      model::BasicBlock &CurrentBlock = Function.CFG[JumpTargetAddress];
      CurrentBlock.End = Source;
      CurrentBlock.Name = JumpTargetBB->getName();
      auto SuccessorsInserter = CurrentBlock.Successors.batch_insert();

      if (EdgeType == FET::DirectBranch) {
        // Handle direct branch
        auto Successors = GCBI.getSuccessors(BB);
        for (const MetaAddress &Destination : Successors.Addresses)
          SuccessorsInserter.insert(FunctionEdge{ Destination, EdgeType });

      } else if (EdgeType == FET::FakeFunctionReturn) {
        // Handle fake function return
        auto [First, Last] = FunctionSummary.FakeReturns.equal_range(BB);
        revng_assert(First != Last);
        for (const auto &[_, Destination] : make_range(First, Last))
          SuccessorsInserter.insert(FunctionEdge{ Destination, EdgeType });

      } else if (IsCall) {
        // Handle call
        llvm::BasicBlock *Successor = BB->getSingleSuccessor();
        MetaAddress Destination = MetaAddress::invalid();
        if (Successor != nullptr)
          Destination = getBasicBlockPC(Successor);

        // Record the edge in the CFG

        auto &Edge = SuccessorsInserter.insert({ Destination, EdgeType });

        bool Found = false;
        for (const FunctionsSummary::CallSiteDescription &CSD :
             FunctionSummary.CallSites) {
          if (not CSD.Call->isTerminator() or CSD.Call->getParent() != BB)
            continue;

          revng_assert(not Found);
          Found = true;
          auto Inserter = Edge.Registers.batch_insert();
          for (auto &[CSV, FCRD] : CSD.RegisterSlots) {
            auto ID = ABIRegister::fromCSVName(CSV->getName(), GCBI.arch());
            if (ID == model::Register::Invalid)
              continue;
            FunctionABIRegister TheRegister(ID);
            TheRegister.Argument = toRegisterState(FCRD.Argument);
            TheRegister.ReturnValue = toRegisterState(FCRD.ReturnValue);
            Inserter.insert(TheRegister);
          }
        }
        revng_assert(Found);

      } else {
        // Handle other successors
        llvm::BasicBlock *Successor = BB->getSingleSuccessor();
        MetaAddress Destination = MetaAddress::invalid();
        if (Successor != nullptr)
          Destination = getBasicBlockPC(Successor);

        // Record the edge in the CFG
        SuccessorsInserter.insert(FunctionEdge{ Destination, EdgeType });
      }
    }
  }

  revng_check(TheBinary.verify());
}

template<class FunctionOracle>
FuncSummary CFEPAnalyzer<FunctionOracle>::analyze(
  const std::vector<llvm::GlobalVariable *> &ABIRegs,
  llvm::BasicBlock *Entry) {
  using namespace llvm;

  Value *RA = nullptr;
  const auto *PCH = GCBI->programCounterHandler();
  Function *OutFunc = createDisposableFunction(Entry);

  ABIAnalyses::analyzeOutlinedFunction(OutFunc, *GCBI);
  // Identify the callee-saved registers and tell if a function is jumping
  // to the return address. To achieve this, we craft the IR by loading the PC,
  // the SP, as well as the ABI registers CSVs at function entry and end.
  IRBuilder<> Builder(&OutFunc->getEntryBlock().front());
  auto *IntPtrTy = GCBI->spReg()->getType();
  auto *IntTy = GCBI->spReg()->getType()->getElementType();

  auto *SP = Builder.CreateLoad(
    M.getGlobalVariable(GCBI->spReg()->getName(), true));
  auto *SPPtr = Builder.CreateIntToPtr(SP, IntPtrTy);
  auto *GEPI = Builder.CreateGEP(IntTy, SPPtr, ConstantInt::get(IntTy, 0));

  switch (GCBI->arch()) {
  case Triple::arm: {
    constexpr auto *LinkRegister = "r14";
    RA = Builder.CreateLoad(M.getGlobalVariable(LinkRegister, true));
    break;
  }

  case Triple::aarch64: {
    constexpr auto *LinkRegister = "x30";
    RA = Builder.CreateLoad(M.getGlobalVariable(LinkRegister, true));
    break;
  };

  case Triple::mips:
  case Triple::mipsel: {
    constexpr auto *ReturnRegister = "ra";
    RA = Builder.CreateLoad(M.getGlobalVariable(ReturnRegister, true));
    break;
  }

  case Triple::systemz: {
    constexpr auto *ReturnRegister = "r14";
    RA = Builder.CreateLoad(M.getGlobalVariable(ReturnRegister, true));
    break;
  }

  case Triple::x86:
  case Triple::x86_64: {
    RA = Builder.CreateLoad(SPPtr);
    break;
  }

  default:
    revng_abort("Unsupported architecture");
  }

  std::array<Value *, 4> DissectedPC = PCH->dissectJumpablePC(Builder,
                                                              RA,
                                                              GCBI->arch());

  auto *PCI = MetaAddress::composeIntegerPC(Builder,
                                            DissectedPC[0],
                                            DissectedPC[1],
                                            DissectedPC[2],
                                            DissectedPC[3]);

  SmallVector<Value *, 8> CSVI;
  for (auto *CSR : ABIRegs) {
    auto *V = Builder.CreateLoad(M.getGlobalVariable(CSR->getName(), true));
    CSVI.emplace_back(V);
  }

  Instruction *Last = nullptr;
  SmallVector<Instruction *, 4> Returns;
  for (auto &BB : *OutFunc)
    if (auto *Ret = dyn_cast<ReturnInst>(&BB.back()))
      Last = Ret;

  if (Last) {
    for (auto *Pred : predecessors(Last->getParent())) {
      auto *Term = Pred->getTerminator();
      Builder.SetInsertPoint(Term);

      auto *PCE = PCH->composeIntegerPC(Builder);

      SmallVector<Value *, 8> CSVE;
      for (auto *CSR : ABIRegs) {
        auto *V = Builder.CreateLoad(M.getGlobalVariable(CSR->getName(), true));
        CSVE.emplace_back(V);
      }

      SP = Builder.CreateLoad(
        M.getGlobalVariable(GCBI->spReg()->getName(), true));
      SPPtr = Builder.CreateIntToPtr(SP, IntPtrTy);
      auto *GEPE = Builder.CreateGEP(IntTy, SPPtr, ConstantInt::get(IntTy, 0));

      auto *SPI = Builder.CreatePtrToInt(GEPI, IntTy);
      auto *SPE = Builder.CreatePtrToInt(GEPE, IntTy);

      // Compute the difference between the return address and the program
      // counter.
      auto *IsRet = Builder.CreateSub(PCE, PCI);

      // Compute the difference between the stack pointer values to evaluate the
      // stack height, and the difference between the CSR values as well.
      // Functions leaving the stack pointer higher than it was at function
      // entry (i.e., in an irregular state) are marked as fake functions.
      auto *SPDiff = Builder.CreateSub(SPE, SPI);

      Type *RetTy = Type::getInt128Ty(Context);
      SmallVector<Type *, 8> ArgTypes = { RetTy, IntTy };
      SmallVector<Value *, 8> ArgValues = { IsRet, SPDiff };
      for (unsigned I = 0; I < CSVI.size(); ++I) {
        auto *V = Builder.CreateSub(CSVE[I], CSVI[I]);
        ArgTypes.emplace_back(IntTy);
        ArgValues.emplace_back(V);
      }

      OFPIndirectBranchInfo.addFnAttribute(Attribute::NoUnwind);
      OFPIndirectBranchInfo.addFnAttribute(Attribute::NoReturn);

      auto *OpqFunc = OFPIndirectBranchInfo
                        .get(Last->getParent()->getParent()->getName(),
                             IntTy,
                             ArgTypes,
                             "indirect_branch_info");

      // Is jumping to the return address? What is the stack height?
      // What are the callee-saved registers? Where do we come from?
      auto *Call = Builder.CreateCall(OpqFunc, ArgValues);
      Builder.CreateUnreachable();

      IndirectBranchInfoCalls.emplace_back(Call);
      Returns.emplace_back(Term);
    }
  }

  for (auto *I : Returns)
    I->eraseFromParent();

  for (auto *I : IndirectBranchInfoCalls) {
    auto *NewPCCall = reverseNewPCTraversal(I);
    MetaAddress NewPC = GCBI::getPCFromNewPC(NewPCCall);
    Builder.SetInsertPoint(NewPCCall);
    PCH->setPC(Builder, NewPC);
  }

  // Run optimization passes over the disposable function
  {
    FunctionPassManager FPM;

    // First stage: simplify the IR and resolve redundant
    // expressions in order to compute the stack height.
    FPM.addPass(RemoveNewPCCallsPass());
    FPM.addPass(RemoveHelperCallsPass());
    FPM.addPass(PromotePass());
    FPM.addPass(JumpThreadingPass());
    FPM.addPass(UnreachableBlockElimPass());
    FPM.addPass(InstCombinePass(false));
    FPM.addPass(EarlyCSEPass(true));
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(MergedLoadStoreMotionPass());
    FPM.addPass(GVN());

    // Second stage: add alias analysis info, perform the GEP
    // transformation and promote CSVs to local variables.
    // Another round of InstCombine and CSE may be necessary
    // to take more optimization opportunities.
    FPM.addPass(InstCombinePass(false));
    FPM.addPass(CSVAliasAnalysisPass<true>());
    FPM.addPass(PromoteGlobalToLocalPass());
    FPM.addPass(InstCombinePass(false));
    FPM.addPass(EarlyCSEPass(true));

    FunctionAnalysisManager FAM;
    FAM.registerPass([] {
      AAManager AA;
      AA.registerFunctionAnalysis<BasicAA>();
      AA.registerFunctionAnalysis<ScopedNoAliasAA>();

      return AA;
    });

    ModuleAnalysisManager MAM;
    MAM.registerPass([&] { return GeneratedCodeBasicInfoAnalysis(); });
    FAM.registerPass([&] { return GeneratedCodeBasicInfoAnalysis(); });
    FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

    PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);
    PB.registerModuleAnalyses(MAM);

    FPM.run(*OutFunc, FAM);
  }

  FuncSummary FunctionInfo = milkResults(ABIRegs, OutFunc);

  // Is the disposable function a FakeFunction? If so, replace
  // the broken returns with valid return instruction, so that
  // it can be inlined.
  if (FunctionInfo.Type == FunctionTypeValue::Fake) {
    for (auto &BB : *OutFunc) {
      // If it is a FakeFunction, it must have broken rets only.
      if (auto *Term = dyn_cast<UnreachableInst>(&BB.back())) {
        ReturnInst *Ret = ReturnInst::Create(Context);
        ReplaceInstWithInst(Term, Ret);
      }
    }
    // WIP: Fake = createDisposable(Entry);
    FunctionInfo.FakeFunction = OutFunc;
  } else {
    throwDisposableFunction(OutFunc);
  }

  return FunctionInfo;
}

template<class FunctionOracle>
FuncSummary CFEPAnalyzer<FunctionOracle>::milkResults(
  const std::vector<llvm::GlobalVariable *> &ABIRegs,
  llvm::Function *F) {
  using namespace llvm;

  SmallVector<std::pair<CallInst *, uint64_t>, 4> MaybeReturns;
  SmallSet<std::pair<CallInst *, FunctionEdgeTypeValue>, 4> Result;
  std::set<GlobalVariable *> CalleeSavedRegs(ABIRegs.begin(), ABIRegs.end());
  std::set<GlobalVariable *> ClobberedRegs(ABIRegs.begin(), ABIRegs.end());

  for (auto *I : IndirectBranchInfoCalls) {
    if (isa<ConstantInt>(I->getArgOperand(0))) {
      if (isa<ConstantInt>(I->getArgOperand(1))) {
        uint64_t FSO = cast<ConstantInt>(I->getArgOperand(1))->getZExtValue();
        if (FSO >= 0)
          MaybeReturns.emplace_back(I, FSO);
      } else {
        Result.insert({ I, FunctionEdgeTypeValue::BrokenReturn });
      }
    }
  }

  // Elect a final stack offset.
  auto WinFSO = [&MaybeReturns]() -> std::optional<uint64_t> {
    auto It = std::min_element(MaybeReturns.begin(),
                               MaybeReturns.end(),
                               [](const auto &LHS, const auto &RHS) {
                                 return LHS.second < RHS.second;
                               });
    if (It == MaybeReturns.end())
      return {};
    return It->second;
  }();

  // Did we find a valid return instruction?
  for (const auto &E : MaybeReturns) {
    auto *CI = E.first;
    std::set<GlobalVariable *> TempCSR;
    if (E.second == *WinFSO) {
      Result.insert({ E.first, FunctionEdgeTypeValue::Return });
      unsigned NumArgs = CI->getNumArgOperands();
      if (NumArgs > 2) {
        for (unsigned Idx = 2; Idx < NumArgs; ++Idx) {
          if (isa<ConstantInt>(CI->getArgOperand(Idx))) {
            TempCSR.insert(ABIRegs[Idx - 2]);
          }
        }
      }

      std::erase_if(CalleeSavedRegs, [&](const auto &E) {
        return TempCSR.find(E) == TempCSR.end();
      });
    } else {
      Result.insert({ E.first, FunctionEdgeTypeValue::BrokenReturn });
    }
  }

  // Neither a return nor a broken return?
  for (auto *I : IndirectBranchInfoCalls) {
    if (isa<ConstantInt>(I->getArgOperand(0)))
      continue;

    if (WinFSO.has_value() && (isa<ConstantInt>(I->getArgOperand(1)))
        && (cast<ConstantInt>(I->getArgOperand(1))->getZExtValue() == *WinFSO))
      Result.insert({ I, FunctionEdgeTypeValue::IndirectTailCall });
    else
      Result.insert({ I, FunctionEdgeTypeValue::LongJmp });
  }

  bool FoundRet, FoundBrokenRet;
  FoundRet = FoundBrokenRet = false;
  for (const auto &E : Result) {
    if (E.second == FunctionEdgeTypeValue::Return)
      FoundRet = true;
    if (E.second == FunctionEdgeTypeValue::BrokenReturn)
      FoundBrokenRet = true;
  }

  FunctionTypeValue Type;
  if (FoundRet) {
    Type = FunctionTypeValue::Regular;
  } else if (FoundBrokenRet) {
    Type = FunctionTypeValue::Fake;
  } else {
    Type = FunctionTypeValue::NoReturn;
  }

  // Finally retrieve the clobbered registers and return them.
  std::erase_if(ClobberedRegs, [&](const auto &E) {
    return CalleeSavedRegs.find(E) != CalleeSavedRegs.end();
  });

  return FuncSummary(Type, ClobberedRegs, Result, WinFSO);
}

template<class FunctionOracle>
llvm::BasicBlock *
CFEPAnalyzer<FunctionOracle>::integrateFunctionCallee(llvm::BasicBlock *BB) {
  using namespace llvm;

  // In case the basic block has a call-site, the function call is
  // replaced with i) hooks that facilitate the ABI analysis ii) summary
  // of the registers clobbered by that function. The newly created basic block
  // is returned so that it can be added to the ValueToValueMap.
  BasicBlock *Split = nullptr;
  auto *Term = BB->getTerminator();
  auto *Call = getFunctionCall(Term);

  // If the successor is null, we are dealing with an indirect call.
  auto *Next = getFunctionCallCallee(Term);

  // What are the registers clobbered by the callee?
  const auto &ClobberedRegs = Oracle.getRegistersClobbered(Next);

  // What is the function type of the callee?
  FunctionTypeValue Ty = Oracle.getFunctionType(Next);

  switch (Ty) {
  case FunctionTypeValue::Regular: {
    BranchInst *Br = BranchInst::Create(getFallthrough(Term));
    ReplaceInstWithInst(Term, Br);

    auto *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(Context), false);
    auto *PreHookOpqF = OFPHooksFunctionCall.get("__precall_pure_function",
                                                 FTy,
                                                 "precall_hook");
    auto *Last = CallInst::Create(PreHookOpqF, "", Br);

    Split = BB->splitBasicBlock(Last->getPrevNode(),
                                BB->getName() + Twine("__summary"));

    // Prevent the store instructions from being optimized out by storing
    // the rval of a call to an opaque function into the clobbered registers.
    OFPRegistersClobbered.addFnAttribute(Attribute::ReadOnly);
    OFPRegistersClobbered.addFnAttribute(Attribute::NoUnwind);

    for (GlobalVariable *Reg : ClobberedRegs) {
      auto *CSVTy = Reg->getType()->getPointerElementType();
      auto *RegsClobbOpqF = OFPRegistersClobbered
                              .get(BB->getParent()->getName(),
                                   CSVTy,
                                   {},
                                   "registers_clobbered");
      new StoreInst(CallInst::Create(RegsClobbOpqF, "", Br), Reg, Br);
    }

    auto *PostHookOpqF = OFPHooksFunctionCall.get("__postcall_pure_function",
                                                  FTy,
                                                  "postcall_hook");
    CallInst::Create(PostHookOpqF, "", Split->getTerminator());
    break;
  }

  case FunctionTypeValue::NoReturn: {
    ReturnInst *Ret = ReturnInst::Create(Context);
    ReplaceInstWithInst(Term, Ret);
    break;
  }

  case FunctionTypeValue::Fake: {
    BranchInst *Br = BranchInst::Create(getFallthrough(Term));

    // Get the fake function by its entry basic block.
    Function *FakeFunction = Oracle.getFakeFunction(Next);

    // Must have already been analyzed.
    revng_assert(FakeFunction != nullptr);

    // If possible, inline the fake function.
    auto *CI = CallInst::Create(FakeFunction, "", Term);
    InlineFunctionInfo IFI;
    bool Status = InlineFunction(llvm::CallSite(CI), IFI, nullptr, true);

    // Still need to replace the successor with the fall-through.
    ReplaceInstWithInst(Term, Br);
    break;
  }

  default:
    revng_abort();
  }

  Call->eraseFromParent();
  return Split;
}

template<class FunctionOracle>
llvm::Function *CFEPAnalyzer<FunctionOracle>::createDisposableFunction(
  llvm::BasicBlock *Entry) {
  using namespace llvm;
  Function *Root = Entry->getParent();

  OnceQueue<BasicBlock *> Queue;
  std::vector<BasicBlock *> BlocksToClone;
  Queue.insert(Entry);

  while (!Queue.empty()) {
    BasicBlock *Current = Queue.pop();

    BlocksToClone.emplace_back(Current);

    if (isFunctionCall(Current)) {
      auto *Succ = getFallthrough(Current);
      revng_assert(Succ != nullptr);
      Queue.insert(Succ);
      continue;
    }

    for (auto *Succ : successors(Current)) {
      if (!GCBI::isPartOfRootDispatcher(Succ))
        Queue.insert(Succ);
    }
  }

  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> BlocksToExtract;

  for (const auto &BB : BlocksToClone) {
    auto *Cloned = CloneBasicBlock(BB, VMap, Twine("__cloned"), Root);

    BasicBlock *New = nullptr;
    if (isFunctionCall(Cloned)) {
      New = integrateFunctionCallee(Cloned);
      VMap[Cloned] = New;
    }

    VMap[BB] = Cloned;
    BlocksToExtract.emplace_back(Cloned);
    if (New)
      BlocksToExtract.emplace_back(New);
  }

  remapInstructionsInBlocks(BlocksToExtract, VMap);

  CodeExtractorAnalysisCache CEAC(*Root);
  Function *OutlinedFunction = CodeExtractor(BlocksToExtract)
                                 .extractCodeRegion(CEAC);
                                 
  revng_assert(OutlinedFunction != nullptr);
  revng_assert(OutlinedFunction->arg_size() == 0);
  revng_assert(OutlinedFunction->getReturnType()->isVoidTy());

  cast<Instruction>(*OutlinedFunction->user_begin())
    ->getParent()
    ->eraseFromParent();

  return OutlinedFunction;
}

template<class FunctionOracle>
void CFEPAnalyzer<FunctionOracle>::throwDisposableFunction(llvm::Function *F) {
  revng_assert(F->use_empty()
               && "Failed to remove all users of the outlined function.");
  F->eraseFromParent();
  F = nullptr;
}

bool StackAnalysis::runOnModule(Module &M) {
  Function &F = *M.getFunction("root");

  revng_log(PassesLog, "Starting StackAnalysis");

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  auto &LMP = getAnalysis<LoadModelPass>();

  // The stack analysis works function-wise. We consider two sets of functions:
  // first (Force == true) those that are highly likely to be real functions
  // (i.e., they have a direct call) and then (Force == false) all the remaining
  // candidates whose entry point is not included in any function of the first
  // set.

  struct CFEP {
    CFEP(BasicBlock *Entry, bool Force) : Entry(Entry), Force(Force) {}

    BasicBlock *Entry;
    bool Force;
  };
  std::vector<CFEP> Functions;

  // Register all the Candidate Function Entry Points
  for (BasicBlock &BB : F) {

    if (GCBI.getType(&BB) != BlockType::JumpTargetBlock)
      continue;

    uint32_t Reasons = GCBI.getJTReasons(&BB);
    bool IsFunctionSymbol = hasReason(Reasons, JTReason::FunctionSymbol);
    bool IsCallee = hasReason(Reasons, JTReason::Callee);
    bool IsUnusedGlobalData = hasReason(Reasons, JTReason::UnusedGlobalData);
    bool IsMemoryStore = hasReason(Reasons, JTReason::MemoryStore);
    bool IsPCStore = hasReason(Reasons, JTReason::PCStore);
    bool IsReturnAddress = hasReason(Reasons, JTReason::ReturnAddress);
    bool IsLoadAddress = hasReason(Reasons, JTReason::LoadAddress);

    if (IsFunctionSymbol or IsCallee) {
      // Called addresses are a strong hint
      Functions.emplace_back(&BB, true);
    } else if (not IsLoadAddress
               and (IsUnusedGlobalData
                    || (IsMemoryStore and not IsPCStore
                        and not IsReturnAddress))) {
      // TODO: keep IsReturnAddress?
      // Consider addresses found in global data that have not been used or
      // addresses that are not return addresses and do not end up in the PC
      // directly.
      Functions.emplace_back(&BB, false);
    }
  }

  for (CFEP &Function : Functions) {
    revng_log(CFEPLog,
              getName(Function.Entry) << (Function.Force ? " (forced)" : ""));
  }

  // Collect all the ABI registers.
  std::vector<llvm::GlobalVariable *> ABIRegisters;
  for (auto *CSV : GCBI.abiRegisters())
    if (CSV && !(GCBI.isSPReg(CSV)))
      ABIRegisters.emplace_back(CSV);

  // Isolate function starting from the collected entrypoints
  // and analyze them.
  using CFEPA = CFEPAnalyzer<FunctionProperties>;
  FunctionProperties Properties;

  for (CFEP &Function :
       llvm::make_range(Functions.rbegin(), Functions.rend())) {
    CFEPA Analyzer(M, &GCBI, Properties);
    Properties.registerFunc(Function.Entry,
                            Analyzer.analyze(ABIRegisters, Function.Entry));
  }

  // Still OK?
  revng_assert(llvm::verifyModule(M, &llvm::dbgs()) == false);

  // Initialize the cache where all the results will be accumulated
  Cache TheCache(&F, &GCBI);

  // Pool where the final results will be collected
  ResultsPool Results;

  // First analyze all the `Force`d functions (i.e., with an explicit direct
  // call)
  for (CFEP &Function : Functions) {
    if (Function.Force) {
      auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
      InterproceduralAnalysis SA(TheCache, GCBI);
      SA.run(Function.Entry, Results);
    }
  }

  // Now analyze all the remaining candidates which are not already part of
  // another function
  std::set<BasicBlock *> Visited = Results.visitedBlocks();
  for (CFEP &Function : Functions) {
    if (not Function.Force and Visited.count(Function.Entry) == 0) {
      auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
      InterproceduralAnalysis SA(TheCache, GCBI);
      SA.run(Function.Entry, Results);
    }
  }

  for (CFEP &Function : Functions) {
    using IFS = IntraproceduralFunctionSummary;
    BasicBlock *Entry = Function.Entry;
    llvm::Optional<const IFS *> Cached = TheCache.get(Entry);
    revng_assert(Cached or TheCache.isFakeFunction(Entry));

    // Has this function been analyzed already? If so, only now we register it
    // in the ResultsPool.
    FunctionType::Values Type;
    if (TheCache.isFakeFunction(Entry))
      Type = FunctionType::Fake;
    else if (TheCache.isNoReturnFunction(Entry))
      Type = FunctionType::NoReturn;
    else
      Type = FunctionType::Regular;

    // Regular functions need to be composed by at least a basic block
    if (Cached) {
      const IFS *Summary = *Cached;
      if (Type == FunctionType::Regular)
        revng_assert(Summary->BranchesType.size() != 0);

      Results.registerFunction(Entry, Type, Summary);
    } else {
      Results.registerFunction(Entry, Type, nullptr);
    }
  }

  GrandResult = Results.finalize(&M, &TheCache);

  if (ClobberedLog.isEnabled()) {
    for (auto &P : GrandResult.Functions) {
      ClobberedLog << getName(P.first) << ":";
      for (const llvm::GlobalVariable *CSV : P.second.ClobberedRegisters)
        ClobberedLog << " " << CSV->getName().data();
      ClobberedLog << DoLog;
    }
  }

  if (StackAnalysisLog.isEnabled()) {
    std::stringstream Output;
    GrandResult.dump(&M, Output);
    TextRepresentation = Output.str();
    revng_log(StackAnalysisLog, TextRepresentation);
  }

  revng_log(PassesLog, "Ending StackAnalysis");

  if (ABIAnalysisOutputPath.getNumOccurrences() == 1) {
    std::ofstream Output;
    serialize(pathToStream(ABIAnalysisOutputPath, Output));
  }

#if 0
  // TODO: fix assert
  commitToModel(GCBI, &F, GrandResult, LMP.getWriteableModel());
#endif

  return false;
}

void StackAnalysis::serializeMetadata(Function &F,
                                      GeneratedCodeBasicInfo &GCBI) {
  using namespace llvm;

  const FunctionsSummary &Summary = GrandResult;

  LLVMContext &Context = getContext(&F);
  QuickMetadata QMD(Context);

  // Temporary data structure so we can set all the `revng.func.member.of` in a
  // single shot at the end
  std::map<Instruction *, std::vector<Metadata *>> MemberOf;

  // Loop over all the detected functions
  for (const auto &P : Summary.Functions) {
    BasicBlock *Entry = P.first;
    const FunctionsSummary::FunctionDescription &Function = P.second;

    if (Entry == nullptr or Function.BasicBlocks.size() == 0)
      continue;

    MetaAddress EntryPC = getBasicBlockPC(Entry);

    //
    // Add `revng.func.entry`:
    // {
    //   name,
    //   address,
    //   type,
    //   { clobbered csv, ... },
    //   { { csv, argument, return value }, ... }
    // }
    //
    auto TypeMD = QMD.get(FunctionType::getName(Function.Type));

    // Clobbered registers metadata
    std::vector<Metadata *> ClobberedMDs;
    for (GlobalVariable *ClobberedCSV : Function.ClobberedRegisters) {
      if (not GCBI.isServiceRegister(ClobberedCSV))
        ClobberedMDs.push_back(QMD.get(ClobberedCSV));
    }

    // Register slots metadata
    std::vector<Metadata *> SlotMDs;
    for (auto &P : Function.RegisterSlots) {
      if (GCBI.isServiceRegister(P.first))
        continue;

      auto *CSV = QMD.get(P.first);
      auto *Argument = QMD.get(P.second.Argument.valueName());
      auto *ReturnValue = QMD.get(P.second.ReturnValue.valueName());
      SlotMDs.push_back(QMD.tuple({ CSV, Argument, ReturnValue }));
    }

    // Create revng.func.entry metadata
    MDTuple *FunctionMD = QMD.tuple({ QMD.get(getName(Entry)),
                                      QMD.get(GCBI.toConstant(EntryPC)),
                                      TypeMD,
                                      QMD.tuple(ClobberedMDs),
                                      QMD.tuple(SlotMDs) });
    Entry->getTerminator()->setMetadata("revng.func.entry", FunctionMD);

    //
    // Create func.call
    //
    for (const FunctionsSummary::CallSiteDescription &CallSite :
         Function.CallSites) {
      Instruction *Call = CallSite.Call;

      // Register slots metadata
      std::vector<Metadata *> SlotMDs;
      for (auto &P : CallSite.RegisterSlots) {
        if (GCBI.isServiceRegister(P.first))
          continue;

        auto *CSV = QMD.get(P.first);
        auto *Argument = QMD.get(P.second.Argument.valueName());
        auto *ReturnValue = QMD.get(P.second.ReturnValue.valueName());
        SlotMDs.push_back(QMD.tuple({ CSV, Argument, ReturnValue }));
      }

      Call->setMetadata("func.call", QMD.tuple(QMD.tuple(SlotMDs)));
    }

    //
    // Create revng.func.member.of
    //

    // Loop over all the basic blocks composing the function
    for (const auto &P : Function.BasicBlocks) {
      BasicBlock *BB = P.first;
      BranchType::Values Type = P.second;

      auto *Pair = QMD.tuple({ FunctionMD, QMD.get(getName(Type)) });

      // Register that this block is associated to this function
      MemberOf[BB->getTerminator()].push_back(Pair);
    }
  }

  // Apply `revng.func.member.of`
  for (auto &P : MemberOf)
    P.first->setMetadata("revng.func.member.of", QMD.tuple(P.second));
}

} // namespace StackAnalysis

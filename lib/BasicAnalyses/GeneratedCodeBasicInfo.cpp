/// \file GeneratedCodeBasicInfo.cpp
/// Implements the GeneratedCodeBasicInfo pass which provides basic information
/// about the translated code (e.g., which CSV is the PC).

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <queue>
#include <set>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"

using namespace llvm;

AnalysisKey GeneratedCodeBasicInfoAnalysis::Key;

char GeneratedCodeBasicInfoWrapperPass::ID = 0;
using RegisterGCBI = RegisterPass<GeneratedCodeBasicInfoWrapperPass>;
static RegisterGCBI X("gcbi", "Generated Code Basic Info", true, true);

void GeneratedCodeBasicInfo::run(Module &M) {
  RootFunction = M.getFunction("root");
  NewPC = M.getFunction("newpc");

  revng_log(PassesLog, "Starting GeneratedCodeBasicInfo");

  using namespace model::Architecture;
  auto Architecture = Binary->Architecture();
  PC = M.getGlobalVariable(getPCCSVName(Architecture), true);
  SP = M.getGlobalVariable(getCSVName(getStackPointer(Architecture)), true);
  auto ReturnAddressRegister = getReturnAddressRegister(Architecture);
  if (ReturnAddressRegister != model::Register::Invalid)
    RA = M.getGlobalVariable(getCSVName(ReturnAddressRegister), true);

  for (model::Register::Values Register : registers(Architecture)) {
    GlobalVariable *CSV = M.getGlobalVariable(getCSVName(Register), true);
    ABIRegisters.push_back(CSV);
    ABIRegistersSet.insert(CSV);
  }

  Type *PCType = PC->getValueType();
  PCRegSize = M.getDataLayout().getTypeAllocSize(PCType);

  for (GlobalVariable &CSV : FunctionTags::CSV.globals(&M))
    CSVs.push_back(&CSV);

  revng_log(PassesLog, "Ending GeneratedCodeBasicInfo");
}

void GeneratedCodeBasicInfo::parseRoot() {
  revng_assert(RootFunction != nullptr);
  revng_assert(not RootFunction->isDeclaration());

  if (RootParsed)
    return;
  RootParsed = true;

  for (BasicBlock &BB : *RootFunction) {
    if (!BB.empty()) {
      switch (getType(&BB)) {
      case BlockType::RootDispatcherBlock:
        revng_assert(Dispatcher == nullptr);
        Dispatcher = &BB;
        break;

      case BlockType::DispatcherFailureBlock:
        revng_assert(DispatcherFail == nullptr);
        DispatcherFail = &BB;
        break;

      case BlockType::AnyPCBlock:
        revng_assert(AnyPC == nullptr);
        AnyPC = &BB;
        break;

      case BlockType::UnexpectedPCBlock:
        revng_assert(UnexpectedPC == nullptr);
        UnexpectedPC = &BB;
        break;

      case BlockType::JumpTargetBlock: {
        auto *Call = cast<CallInst>(&*BB.begin());
        revng_assert(Call->getCalledFunction() == NewPC);
        JumpTargets[addressFromNewPC(Call)] = &BB;
        break;
      }
      case BlockType::RootDispatcherHelperBlock:
      case BlockType::IndirectBranchDispatcherHelperBlock:
      case BlockType::EntryPoint:
      case BlockType::ExternalJumpsHandlerBlock:
      case BlockType::TranslatedBlock:
        // Nothing to do here
        break;
      }
    }
  }
}

SmallVector<std::pair<BasicBlock *, bool>, 4>
GeneratedCodeBasicInfo::blocksByPCRange(MetaAddress Start, MetaAddress End) {
  SmallVector<std::pair<BasicBlock *, bool>, 4> Result;

  BasicBlock *StartBB = getBlockAt(Start);

  df_iterator_default_set<BasicBlock *> Visited;
  for (BasicBlock *BB : depth_first_ext(StartBB, Visited)) {
    // Detect if this basic block is a boundary
    enum {
      Unknown,
      Yes,
      No
    } IsBoundary = Unknown;

    auto SuccBegin = succ_begin(BB);
    auto SuccEnd = succ_end(BB);
    if (SuccBegin == SuccEnd) {
      // This basic blocks ends with an `UnreachableInst`
      IsBoundary = Yes;
    } else {
      for (BasicBlock *Successor : make_range(SuccBegin, SuccEnd)) {

        // Ignore unexpectedpc
        if (getType(Successor) == BlockType::UnexpectedPCBlock)
          continue;

        auto SuccessorMA = getBasicBlockAddress(Successor);
        if (not isPartOfRootDispatcher(Successor)
            and (SuccessorMA.isInvalid()
                 or (SuccessorMA.address() >= Start.address()
                     and SuccessorMA.address() < End.address()))) {
          revng_assert(IsBoundary != Yes);
          IsBoundary = No;
        } else {
          revng_assert(IsBoundary != No);
          IsBoundary = Yes;
          Visited.insert(Successor);
        }
      }
    }

    revng_assert(IsBoundary != Unknown);

    Result.emplace_back(BB, IsBoundary == Yes);
  }

  return Result;
}

GeneratedCodeBasicInfo
GeneratedCodeBasicInfoAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  auto &LMA = MAM.getResult<LoadModelAnalysis>(M);
  GeneratedCodeBasicInfo GCBI(*LMA.getReadOnlyModel());
  GCBI.run(M);
  return GCBI;
}

GeneratedCodeBasicInfo
GeneratedCodeBasicInfoAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  auto &LMA = FAM.getResult<LoadModelAnalysis>(F);
  GeneratedCodeBasicInfo GCBI(*LMA.getReadOnlyModel());
  GCBI.run(*F.getParent());
  return GCBI;
}

bool GeneratedCodeBasicInfoWrapperPass::runOnModule(Module &M) {
  auto &LMA = getAnalysis<LoadModelWrapperPass>().get();
  GCBI.reset(new GeneratedCodeBasicInfo(*LMA.getReadOnlyModel()));
  GCBI->run(M);
  return false;
}

void GeneratedCodeBasicInfoWrapperPass::releaseMemory() {
  GCBI.reset();
}

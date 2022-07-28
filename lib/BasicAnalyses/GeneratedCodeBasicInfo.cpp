/// \file GeneratedCodeBasicInfo.cpp
/// \brief Implements the GeneratedCodeBasicInfo pass which provides basic
///        information about the translated code (e.g., which CSV is the PC).

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <queue>
#include <set>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/Generated/Early/Register.h"
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
  if (NewPC != nullptr)
    MetaAddressStruct = cast<StructType>(NewPC->arg_begin()->getType());

  revng_log(PassesLog, "Starting GeneratedCodeBasicInfo");

  using namespace model::Architecture;
  auto Architecture = Binary->Architecture;
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

  Type *PCType = PC->getType()->getPointerElementType();
  PCRegSize = M.getDataLayout().getTypeAllocSize(PCType);

  QuickMetadata QMD(M.getContext());
  if (auto *NamedMD = M.getNamedMetadata("revng.csv")) {
    auto *Tuple = cast<MDTuple>(NamedMD->getOperand(0));
    for (const MDOperand &Operand : Tuple->operands()) {
      if (Operand.get() == nullptr)
        continue;

      auto *CSV = cast<GlobalVariable>(QMD.extract<Constant *>(Operand.get()));
      CSVs.push_back(CSV);
    }
  }

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
        JumpTargets[MetaAddress::fromConstant(Call->getArgOperand(0))] = &BB;
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

GeneratedCodeBasicInfo::SuccessorsList
GeneratedCodeBasicInfo::getSuccessors(BasicBlock *BB) {
  parseRoot();

  bool IsRoot = BB->getParent() == RootFunction;

  SuccessorsList Result;

  df_iterator_default_set<BasicBlock *> Visited;

  if (IsRoot) {
    Visited.insert(AnyPC);
    Visited.insert(UnexpectedPC);
  }

  for (BasicBlock *Block : depth_first_ext(BB, Visited)) {
    for (BasicBlock *Successor : successors(Block)) {
      revng_assert(Successor != Dispatcher);

      MetaAddress Address = getBasicBlockPC(Successor);
      const auto IBDHB = BlockType::IndirectBranchDispatcherHelperBlock;
      if (Address.isValid()) {
        Visited.insert(Successor);
        Result.Addresses.insert(Address);
      } else if (IsRoot and Successor == AnyPC) {
        Result.AnyPC = true;
      } else if (IsRoot and Successor == UnexpectedPC) {
        Result.UnexpectedPC = true;
      } else if (getType(Successor) == IBDHB) {
        // Ignore
      } else {
        Result.Other = true;
      }
    }
  }

  return Result;
}

SmallVector<std::pair<BasicBlock *, bool>, 4>
GeneratedCodeBasicInfo::blocksByPCRange(MetaAddress Start, MetaAddress End) {
  SmallVector<std::pair<BasicBlock *, bool>, 4> Result;

  BasicBlock *StartBB = getBlockAt(Start);

  df_iterator_default_set<BasicBlock *> Visited;
  for (BasicBlock *BB : depth_first_ext(StartBB, Visited)) {
    // Detect if this basic block is a boundary
    enum { Unknown, Yes, No } IsBoundary = Unknown;

    auto SuccBegin = succ_begin(BB);
    auto SuccEnd = succ_end(BB);
    if (SuccBegin == SuccEnd) {
      // This basic blocks ends with an `UnreachableInst`
      IsBoundary = Yes;
    } else {
      for (BasicBlock *Successor : make_range(SuccBegin, SuccEnd)) {

        // Ignore unexpectedpc
        using GCBI = GeneratedCodeBasicInfo;
        if (getType(Successor) == BlockType::UnexpectedPCBlock)
          continue;

        auto SuccessorMA = GCBI::getPCFromNewPC(Successor);
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

static RecursiveCoroutine<void>
findJumpTarget(llvm::BasicBlock *&Result,
               llvm::BasicBlock *BB,
               std::set<BasicBlock *> &Visited) {
  Visited.insert(BB);

  if (GeneratedCodeBasicInfo::isJumpTarget(BB)) {
    revng_assert(Result == nullptr,
                 "This block leads to multiple jump targets");
    Result = BB;
  } else {
    for (BasicBlock *Predecessor : predecessors(BB)) {
      if (Visited.count(Predecessor) == 0) {
        rc_recur findJumpTarget(Result, Predecessor, Visited);
      }
    }
  }

  rc_return;
}

llvm::BasicBlock *
GeneratedCodeBasicInfo::getJumpTargetBlock(llvm::BasicBlock *BB) {
  BasicBlock *Result = nullptr;
  std::set<BasicBlock *> Visited;
  findJumpTarget(Result, BB, Visited);
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

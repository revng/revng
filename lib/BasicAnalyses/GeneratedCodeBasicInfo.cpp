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

  const char *MDName = "revng.input.architecture";
  NamedMDNode *InputArchMD = M.getOrInsertNamedMetadata(MDName);
  auto *Tuple = dyn_cast<MDTuple>(InputArchMD->getOperand(0));

  QuickMetadata QMD(M.getContext());

  {
    unsigned Index = 0;
    StringRef ArchTypeName = QMD.extract<StringRef>(Tuple, Index++);
    ArchType = Triple::getArchTypeForLLVMName(ArchTypeName);
    InstructionAlignment = QMD.extract<uint32_t>(Tuple, Index++);
    DelaySlotSize = QMD.extract<uint32_t>(Tuple, Index++);
    PC = M.getGlobalVariable(QMD.extract<StringRef>(Tuple, Index++), true);
    SP = M.getGlobalVariable(QMD.extract<StringRef>(Tuple, Index++), true);
    RA = M.getGlobalVariable(QMD.extract<StringRef>(Tuple, Index++), true);
    MinimalFSO = QMD.extract<int64_t>(Tuple, Index++);
    auto Operands = QMD.extract<MDTuple *>(Tuple, Index++)->operands();
    for (const MDOperand &Operand : Operands) {
      StringRef Name = QMD.extract<StringRef>(Operand.get());
      revng_assert(Name != "pc", "PC should not be considered an ABI register");
      GlobalVariable *CSV = M.getGlobalVariable(Name, true);
      ABIRegisters.push_back(CSV);
      ABIRegistersSet.insert(CSV);
    }
  }

  Type *PCType = PC->getType()->getPointerElementType();
  PCRegSize = M.getDataLayout().getTypeAllocSize(PCType);

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

  revng_assert(BB->getParent() == RootFunction);

  SuccessorsList Result;

  df_iterator_default_set<BasicBlock *> Visited;
  Visited.insert(AnyPC);
  Visited.insert(UnexpectedPC);
  for (BasicBlock *Block : depth_first_ext(BB, Visited)) {
    for (BasicBlock *Successor : successors(Block)) {
      revng_assert(Successor != Dispatcher);

      MetaAddress Address = getBasicBlockPC(Successor);
      const auto IBDHB = BlockType::IndirectBranchDispatcherHelperBlock;
      if (Address.isValid()) {
        Visited.insert(Successor);
        Result.Addresses.insert(Address);
      } else if (Successor == AnyPC) {
        Result.AnyPC = true;
      } else if (Successor == UnexpectedPC) {
        Result.UnexpectedPC = true;
      } else if (getType(Successor) == IBDHB) {
        // Ignore
      } else {
        return SuccessorsList::other();
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
        if (GCBI::getType(Successor) == BlockType::UnexpectedPCBlock)
          continue;

        auto SuccessorMA = GCBI::getPCFromNewPC(Successor);
        if (not GCBI::isPartOfRootDispatcher(Successor)
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

llvm::BasicBlock *
GeneratedCodeBasicInfo::getJumpTargetBlock(llvm::BasicBlock *BB) {
  const DominatorTree &DT = getDomTree(BB->getParent());
  auto *Node = DT.getNode(BB);
  revng_assert(Node != nullptr);

  while (Node != nullptr and not isJumpTarget(Node->getBlock())) {
    Node = Node->getIDom();
  }

  if (Node == nullptr)
    return nullptr;
  else
    return Node->getBlock();
}

void GeneratedCodeBasicInfo::initializePCToBlockCache() {
  const DominatorTree &DT = getDomTree(RootFunction);
  for (BasicBlock &BB : *RootFunction) {
    if (not GeneratedCodeBasicInfo::isTranslated(&BB))
      continue;

    auto *DTNode = DT.getNode(&BB);

    // Ignore unreachable basic block
    if (DTNode == nullptr)
      continue;

    while (not GeneratedCodeBasicInfo::isJumpTarget(DTNode->getBlock())) {
      DTNode = DTNode->getIDom();
      revng_assert(DTNode != nullptr);
    }

    PCToBlockCache.insert({ getBasicBlockPC(DTNode->getBlock()), &BB });
  }
}

GeneratedCodeBasicInfo
GeneratedCodeBasicInfoAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  GeneratedCodeBasicInfo GCBI;
  GCBI.run(M);
  return GCBI;
}

GeneratedCodeBasicInfo
GeneratedCodeBasicInfoAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  GeneratedCodeBasicInfo GCBI;
  GCBI.run(*F.getParent());
  return GCBI;
}

bool GeneratedCodeBasicInfoWrapperPass::runOnModule(Module &M) {
  GCBI.reset(new GeneratedCodeBasicInfo());
  GCBI->run(M);
  return false;
}

void GeneratedCodeBasicInfoWrapperPass::releaseMemory() {
  GCBI.reset();
}

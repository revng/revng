/// \file CollectCFG.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/Binary.h"

using namespace llvm;

namespace efa {
class CollectCFG {
private:
  llvm::Module &M;
  GeneratedCodeBasicInfo &GCBI;
  const TupleTree<model::Binary> &Binary;
  CFGAnalyzer &Analyzer;

public:
  CollectCFG(llvm::Module &M,
             GeneratedCodeBasicInfo &GCBI,
             const TupleTree<model::Binary> &Binary,
             CFGAnalyzer &Analyzer) :
    M(M), GCBI(GCBI), Binary(Binary), Analyzer(Analyzer) {}

public:
  void run() {
    auto Result = recoverCFGs();
    serializeFunctionMetadata(Result);
  }

private:
  std::vector<efa::FunctionMetadata> recoverCFGs();

  void
  serializeFunctionMetadata(const std::vector<efa::FunctionMetadata> &CFGs);
};

using CFGVector = vector<FunctionMetadata>;

void CollectCFG::serializeFunctionMetadata(const CFGVector &CFGs) {
  using namespace llvm;
  using llvm::BasicBlock;

  LLVMContext &Context = M.getContext();

  for (const efa::FunctionMetadata &FM : CFGs) {
    FM.verify(*Binary, true);
    BasicBlock *BB = GCBI.getBlockAt(FM.Entry());
    std::string Buffer;
    {
      raw_string_ostream Stream(Buffer);
      serialize(Stream, FM);
    }

    Instruction *Term = BB->getTerminator();
    MDNode *Node = MDNode::get(Context, MDString::get(Context, Buffer));
    Term->setMetadata(FunctionMetadataMDName, Node);
  }
}

std::vector<efa::FunctionMetadata> CollectCFG::recoverCFGs() {
  std::vector<efa::FunctionMetadata> Result;
  for (const auto &Function : Binary->Functions()) {
    auto *Entry = GCBI.getBlockAt(Function.Entry());
    revng_assert(Entry != nullptr);

    // Recover the control-flow graph of the function
    efa::FunctionMetadata New;
    New.Entry() = Function.Entry();
    New.ControlFlowGraph() = std::move(Analyzer.analyze(Entry).CFG);

    revng_assert(New.ControlFlowGraph().count(BasicBlockID(New.Entry())) != 0);

    // Run final steps on the CFG
    New.simplify(*Binary);

    revng_assert(New.ControlFlowGraph().count(BasicBlockID(New.Entry())) != 0);

    Result.emplace_back(std::move(New));
  }
  return Result;
}

bool CollectCFGPass::runOnModule(Module &M) {
  revng_log(PassesLog, "Starting EarlyFunctionAnalysis");

  if (not M.getFunction("root") or M.getFunction("root")->isDeclaration())
    return false;

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &LMP = getAnalysis<LoadModelWrapperPass>().get();

  const TupleTree<model::Binary> &Binary = LMP.getReadOnlyModel();

  FunctionSummaryOracle Oracle;
  importModel(M, GCBI, *Binary, Oracle);

  CFGAnalyzer Analyzer(M, GCBI, Binary, Oracle);

  CollectCFG CFGCollector(M, GCBI, Binary, Analyzer);

  CFGCollector.run();

  return false;
}

char CollectCFGPass::ID = 0;

using CFGCollectionPass = RegisterPass<CollectCFGPass>;
static CFGCollectionPass X("collect-cfg", "CFG Collection Pass", true, true);

} // namespace efa

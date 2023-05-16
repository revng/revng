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
  GeneratedCodeBasicInfo &GCBI;
  const TupleTree<model::Binary> &Binary;
  CFGAnalyzer &Analyzer;

public:
  CollectCFG(GeneratedCodeBasicInfo &GCBI,
             const TupleTree<model::Binary> &Binary,
             CFGAnalyzer &Analyzer) :
    GCBI(GCBI), Binary(Binary), Analyzer(Analyzer) {}

public:
  void run();
};

void CollectCFG::run() {
  for (const auto &Function : Binary->Functions()) {
    auto *Entry = GCBI.getBlockAt(Function.Entry());
    revng_assert(Entry != nullptr);

    // Recover the control-flow graph of the function
    efa::FunctionMetadata New;
    New.Entry() = Function.Entry();
    New.ControlFlowGraph() = std::move(Analyzer.analyze(Entry).CFG);

    revng_assert(New.ControlFlowGraph().contains(BasicBlockID(New.Entry())));

    // Run final steps on the CFG
    New.simplify(*Binary);

    revng_assert(New.ControlFlowGraph().contains(BasicBlockID(New.Entry())));

    New.serialize(GCBI);
  }
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

  CollectCFG CFGCollector(GCBI, Binary, Analyzer);

  CFGCollector.run();

  return false;
}

char CollectCFGPass::ID = 0;

using CFGCollectionPass = RegisterPass<CollectCFGPass>;
static CFGCollectionPass X("collect-cfg", "CFG Collection Pass", true, true);

} // namespace efa

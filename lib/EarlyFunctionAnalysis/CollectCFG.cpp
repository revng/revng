//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"
#include "revng/Model/Binary.h"
#include "revng/Support/CommonOptions.h"
#include "revng/Support/YAMLTraits.h"

namespace revng::pypeline::piperuns {

using FSO = efa::FunctionSummaryOracle;

CollectCFG::CollectCFG(const class Model &Model,
                       llvm::StringRef Config,
                       llvm::StringRef DynamicConfig,
                       LLVMRootContainer &Input,
                       CFGMap &Output) :
  Model(Model),
  Output(Output),
  GCBI(*Model.get().get(), Input.getModule()),
  Oracle(FSO::importBasicPrototypeData(Input.getModule(),
                                       GCBI,
                                       *Model.get().get())),
  Analyzer(Input.getModule(), GCBI, Model.get(), Oracle) {
}

void CollectCFG::runOnFunction(const model::Function &Function) {
  MetaAddress EntryAddress = Function.Entry();
  const model::Binary &Binary = *Model.get().get();

  // Recover the control-flow graph of the function
  TupleTree<efa::FunctionBundle> New;
  efa::ControlFlowGraph &Main = New->MainFunction();
  Main.Entry() = EntryAddress;
  Main.Blocks() = std::move(Analyzer.analyze(EntryAddress).CFG);

  if (DebugNames) {
    auto Function = Binary.Functions().at(EntryAddress);
    Main.Name() = Function.Name();
  }

  if (Main.Blocks().size() > 0)
    revng_assert(Main.Blocks().contains(BasicBlockID(Main.Entry())));

  // Run final steps on the CFG
  Main.simplify(Binary);

  if (Main.Blocks().size() > 0)
    revng_assert(Main.Blocks().contains(BasicBlockID(Main.Entry())));

  Output.getElement(ObjectID(EntryAddress)) = std::move(New);
}

} // namespace revng::pypeline::piperuns

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"
#include "revng-c/ValueManipulationAnalysis/VMAPipeline.h"
#include "revng-c/ValueManipulationAnalysis/ValueManipulationAnalysis.h"

#include "TypeFlowGraph.h"

using namespace llvm;
using namespace vma;

static Logger<> VMALog("vma");
static Logger<> GraphLog("vma-graph");
static Logger<> TimerLogger("vma-timer");

char ValueManipulationAnalysis::ID = 0;
using Register = RegisterPass<ValueManipulationAnalysis>;
static Register X("vma", "Value Manipulation Analysis", false, false);

using VMA = ValueManipulationAnalysis;

/// Simple updater for testing
class ColorMapUpdater : public VMAUpdater {
private:
  VMA::ColorMapT &ColorMap;

public:
  ColorMapUpdater(VMA::ColorMapT &Map) : ColorMap(Map) {}

  void updateWithResults(const vma::TypeFlowGraph *TFG) override {
    // Populate Output Map
    for (const auto *N : TFG->nodes()) {
      revng_assert(N->getCandidates().countValid() <= 1);

      if (N->isValue())
        ColorMap.insert({ N->getValue(), N->getCandidates() });
    }
  }
};

// LLVM Pass
void VMA::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
  AU.setPreservesAll();
}

bool VMA::runOnFunction(Function &F) {
  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Model = *ModelWrapper.getReadOnlyModel();

  revng_log(VMALog, "------ Function: " << F.getName() << " ------");

  ColorMap.clear();

  VMAPipeline VMA(Model);
  VMA.addInitializer(std::make_unique<LLVMInitializer>());
  VMA.setUpdater(std::make_unique<ColorMapUpdater>(ColorMap));
  VMA.enableSolver();

  VMA.run(&F);

  return false;
}

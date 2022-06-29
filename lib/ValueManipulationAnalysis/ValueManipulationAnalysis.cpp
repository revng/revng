//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <cstddef>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"
#include "revng-c/ValueManipulationAnalysis/ValueManipulationAnalysis.h"

#include "Mincut.h"
#include "TypeFlowGraph.h"
#include "TypeFlowNode.h"

using namespace llvm;
using namespace vma;

static Logger<> VMALog("vma");
static Logger<> GraphLog("vma-graph");
static Logger<> TimerLogger("vma-timer");

char ValueManipulationAnalysis::ID = 0;
using Register = RegisterPass<ValueManipulationAnalysis>;
static Register X("vma", "Value Manipulation Analysis", false, false);

using VMA = ValueManipulationAnalysis;

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

  revng_log(VMALog, "------ Function: " << F.getName() << " ------");

  // Color initialization
  ColorMap.clear();
  TypeFlowGraph TG = makeTypeFlowGraphFromFunction(&F);

  // Color propagation
  do {
    propagateColors(TG);
  } while (propagateNumberness(TG));

  // Mincut preprocessing
  makeBidirectional(TG);
  applyMajorityVoting(TG);

  if (GraphLog.isEnabled())
    TG.view();

  // Assign grey nodes
  if (llvm::any_of(TG.nodes(),
                   [](TypeFlowNode *N) { return N->isUndecided(); })) {

    revng_log(VMALog, "Executing mincut");

    std::chrono::steady_clock::time_point Begin;
    if (TimerLogger.isEnabled()) {
      Begin = std::chrono::steady_clock::now();
    }

    minCut(TG);

    if (TimerLogger.isEnabled()) {
      auto End = std::chrono::steady_clock::now();
      auto Dur = End - Begin;
      revng_log(TimerLogger, "total time: " << Dur.count());
      revng_log(TimerLogger, "total dim: " << TG.size());
    }
  } else {
    revng_log(VMALog, "Nothing to assign");
  }

  if (GraphLog.isEnabled())
    TG.view();
  revng_log(VMALog, "Total function cost: " << countCasts(TG));

  // Populate Output Map
  for (const auto *N : TG.nodes()) {
    revng_assert(N->getCandidates().countValid() <= 1);

    if (N->isValue())
      ColorMap.insert({ N->getValue(), N->getCandidates() });
  }

  return false;
}

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng/MFP/MFP.h"

#include "Analyses.h"

namespace ABIAnalyses::RegisterArgumentsOfFunctionCall {
using namespace llvm;
using namespace ABIAnalyses;

std::map<const GlobalVariable *, State>
analyze(const BasicBlock *CallSiteBlock, const GeneratedCodeBasicInfo &GCBI) {
  using MFI = MFIAnalysis<false, CoreLattice>;

  MFI Instance{ { getPostCallHook(CallSiteBlock), GCBI } };
  MFI::LatticeElement InitialValue;
  MFI::LatticeElement ExtremalValue(CoreLattice::ExtremalLatticeElement);

  auto *Start = CallSiteBlock->getUniquePredecessor();
  revng_assert(Start != nullptr, "Call site has multiple predecessors");
  auto
    Results = MFP::getMaximalFixedPoint<MFI, MFI::GT, MFI::LGT>(Instance,
                                                                Start,
                                                                InitialValue,
                                                                ExtremalValue,
                                                                { Start },
                                                                { Start });

  std::map<const GlobalVariable *, State> RegYes;

  const BasicBlock *Entrypoint = &CallSiteBlock->getParent()->getEntryBlock();

  auto It = Results.find(Entrypoint);
  if (It != Results.end()) {
    for (auto &[GV, RegState] : It->second.OutValue) {
      if (RegState == CoreLattice::Yes) {
        RegYes[GV] = State::NoOrDead;
      }
    }
  }

  return RegYes;
}
} // namespace ABIAnalyses::RegisterArgumentsOfFunctionCall

/// \file stackanalysis.cpp
/// \brief Implementation of the stack analysis, which provides information
///        about function boundaries, basic block types, arguments and return
///        values.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <map>
#include <sstream>
#include <vector>

// LLVM includes
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

// Local includes
#include "cache.h"
#include "interprocedural.h"
#include "intraprocedural.h"
#include "ir-helpers.h"
#include "revng/StackAnalysis/stackanalysis.h"

using llvm::BasicBlock;
using llvm::Function;
using llvm::Module;

namespace StackAnalysis {

template<>
char StackAnalysis<true>::ID = 0;
static llvm::RegisterPass<StackAnalysis<true>> X("sa",
                                                 "Stack Analysis Pass",
                                                 true,
                                                 true);

template<>
char StackAnalysis<false>::ID = 0;
static llvm::RegisterPass<StackAnalysis<false>> Y("sab",
                                                  "Stack Analysis Pass with ABI"
                                                  " Analysis",
                                                  true,
                                                  true);

template<bool AnalyzeABI>
bool StackAnalysis<AnalyzeABI>::runOnFunction(Function &F) {

  DBG("passes", { dbg << "Starting StackAnalysis\n"; });

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();
  const Module *M = F.getParent();

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
    if (GCBI.getType(&BB) != JumpTargetBlock)
      continue;

    uint32_t Reasons = GCBI.getJTReasons(&BB);
    bool IsCallee = hasReason(Reasons, JTReason::Callee);
    bool IsUnusedGlobalData = hasReason(Reasons, JTReason::UnusedGlobalData);
    bool IsSETNotToPC = hasReason(Reasons, JTReason::SETNotToPC);
    bool IsSETToPC = hasReason(Reasons, JTReason::SETToPC);
    bool IsReturnAddress = hasReason(Reasons, JTReason::ReturnAddress);
    bool IsLoadAddress = hasReason(Reasons, JTReason::LoadAddress);

    if (IsCallee) {
      // Called addresses are a strong hint
      Functions.emplace_back(&BB, true);
    } else if (not IsLoadAddress
               and (IsUnusedGlobalData
                    || (IsSETNotToPC
                        and not IsSETToPC
                        and not IsReturnAddress))) {
      // TODO: keep IsReturnAddress?
      // Consider addresses found in global data that have not been used in SET
      // or addresses coming from SET that are not return addresses and do not
      // end up in the PC directly.
      Functions.emplace_back(&BB, false);
    }
  }

  // Initialize the cache where all the results will be accumulated
  Cache TheCache(&F);

  // Pool where the final results will be collected
  ResultsPool Results;

  // First analyze all the `Force`d functions (i.e., with an explicit direct
  // call)
  for (CFEP &Function : Functions) {
    if (Function.Force) {
      auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();
      InterproceduralAnalysis SA(TheCache, GCBI, AnalyzeABI);
      SA.run(Function.Entry, Results);
    }
  }

  // Now analyze all the remaining candidates which are not already part of
  // another function
  std::set<BasicBlock *> Visited = Results.visitedBlocks();
  for (CFEP &Function : Functions) {
    if (not Function.Force and Visited.count(Function.Entry) == 0) {
      auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();
      InterproceduralAnalysis SA(TheCache, GCBI, AnalyzeABI);
      SA.run(Function.Entry, Results);
    }
  }

  std::stringstream Output;
  GrandResult = Results.finalize(M);
  GrandResult.dump(M, Output);
  TextRepresentation = Output.str();

  DBG("passes", { dbg << "Ending StackAnalysis\n"; });

  return false;
}

} // namespace StackAnalysis

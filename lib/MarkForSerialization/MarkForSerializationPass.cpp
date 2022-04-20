//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

/// Dataflow analysis to identify which Instructions must be serialized

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/MarkForSerialization/MarkAnalysis.h"
#include "revng-c/MarkForSerialization/MarkForSerializationPass.h"
#include "revng-c/RestructureCFGPass/RestructureCFG.h"
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

Logger<> MarkLog("mark-serialization");

void MarkForSerializationPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<RestructureCFG>();
  AU.setPreservesAll();
}

bool MarkForSerializationPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  // If the `-single-decompilation` option was passed from command line, skip
  // decompilation for all the functions that are not the selected one.
  if (not TargetFunction.empty())
    if (not F.getName().equals(TargetFunction.c_str()))
      return false;

  // Mark instructions for serialization, and write the results in ToSerialize
  ToSerialize = {};
  MarkAnalysis::Analysis Mark(F, ToSerialize);
  Mark.initialize();
  Mark.run();

  return false;
}

char MarkForSerializationPass::ID = 0;

using Register = llvm::RegisterPass<MarkForSerializationPass>;
static Register X("mark-for-serialization",
                  "Pass that marks Instructions for serialization in C",
                  false,
                  false);

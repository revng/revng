/// Collect the function entry points from the callees.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromCalleesPass.h"

using namespace llvm;

static Logger Log("functions-from-callees-collection");

void collectFunctionsFromCallees(Module &M,
                                 GeneratedCodeBasicInfo &GCBI,
                                 model::Binary &Binary) {
  Function &Root = *M.getFunction("root");

  // Static symbols have already been registered during lifting phase. Now
  // register all the other candidate entry points.
  for (BasicBlock &BB : Root) {
    if (getType(&BB) != BlockType::JumpTargetBlock)
      continue;

    MetaAddress Entry = getBasicBlockAddress(getJumpTargetBlock(&BB));
    if (Binary.Functions().contains(Entry))
      continue;

    uint32_t Reasons = GCBI.getJTReasons(&BB);
    bool IsCallee = hasReason(Reasons, JTReason::Callee);

    if (IsCallee) {
      // Create the function
      Binary.Functions()[Entry];
      revng_log(Log, "Found function from callee: " << BB.getName().str());
    }
  }
}

#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <optional>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Pass.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/AAWriterPass.h"
#include "revng/EarlyFunctionAnalysis/IndirectBranchInfoPrinterPass.h"
#include "revng/EarlyFunctionAnalysis/PromoteGlobalToLocalVars.h"
#include "revng/EarlyFunctionAnalysis/SegregateDirectStackAccesses.h"
#include "revng/FunctionCallIdentification/FunctionCallIdentification.h"
#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/OpaqueFunctionsPool.h"

namespace EarlyFunctionAnalysis {

template<bool ShouldAnalyzeABI>
class EarlyFunctionAnalysis : public llvm::ModulePass {
public:
  static char ID;

public:
  EarlyFunctionAnalysis() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool runOnModule(llvm::Module &M) override;
};

template<>
char EarlyFunctionAnalysis<true>::ID;

template<>
char EarlyFunctionAnalysis<false>::ID;

} // namespace EarlyFunctionAnalysis

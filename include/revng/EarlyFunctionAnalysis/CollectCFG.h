#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Pipebox/TupleTreeContainer.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"

namespace revng::pypeline {

class CFGMap
  : public TupleTreeContainer<efa::ControlFlowGraph, Kinds::Function> {
public:
  static constexpr llvm::StringRef Name = "CFGMap";
};

namespace piperuns {

class CollectCFG {
private:
  const class Model &Model;
  CFGMap &Output;

  GeneratedCodeBasicInfo GCBI;
  efa::FunctionSummaryOracle Oracle;
  efa::CFGAnalyzer Analyzer;

public:
  static constexpr llvm::StringRef Name = "collect-cfg";
  using Arguments = TypeList<PipeRunArgument<LLVMRootContainer,
                                             "Input",
                                             "LLVM module to analyze to "
                                             "produce the CFG",
                                             // The root container is
                                             // manipulated to create the
                                             // CFGMap, hence the need to
                                             // declare Access::Read and a
                                             // non-const argument.
                                             Access::Read>,
                             PipeRunArgument<CFGMap,
                                             "Output",
                                             "The produced CFG for each "
                                             "function",
                                             Access::Write>>;

  CollectCFG(const class Model &Model,
             llvm::StringRef Config,
             llvm::StringRef DynamicConfig,
             LLVMRootContainer &Input,
             CFGMap &Output);
  void runOnFunction(const model::Function &TheFunction);
};

} // namespace piperuns

} // namespace revng::pypeline

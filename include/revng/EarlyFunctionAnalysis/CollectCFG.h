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

using CFGMap = TupleTreeContainer<efa::ControlFlowGraph,
                                  Kinds::Function,
                                  "CFGMap">;

namespace piperuns {

namespace detail {

struct GCBIRun {
  GCBIRun(GeneratedCodeBasicInfo &GCBI, llvm::Module &Module) {
    GCBI.run(Module);
  }
};

} // namespace detail

class CollectCFG {
private:
  const class Model &Model;
  CFGMap &Output;

  GeneratedCodeBasicInfo GCBI;
  detail::GCBIRun GCBIRun;
  efa::FunctionSummaryOracle Oracle;
  efa::CFGAnalyzer Analyzer;

public:
  static constexpr llvm::StringRef Name = "CollectCFG";
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

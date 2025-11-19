#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/Pipebox/TupleTreeContainer.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/Pipes/YieldControlFlow.h"

class DissassemblyHelper;

namespace revng::pipes {

class ProcessAssembly {
public:
  static constexpr const auto Name = "process-assembly";

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;

    return { ContractGroup{
      Contract(kinds::Binary, 0, kinds::Binary, 0, InputPreservation::Preserve),
      Contract(kinds::CFG,
               1,
               kinds::FunctionAssemblyInternal,
               2,
               InputPreservation::Preserve) } };
  }

public:
  void run(pipeline::ExecutionContext &Context,
           const BinaryFileContainer &SourceBinary,
           const CFGMap &CFGMap,
           FunctionAssemblyStringMap &OutputAssembly);
};

} // namespace revng::pipes

namespace revng::pypeline {

using AssemblyInternalContainer = TupleTreeContainer<yield::Function,
                                                     Kinds::Function,
                                                     "AssemblyInternalContaine"
                                                     "r">;

namespace piperuns {

class ProcessAssembly {
private:
  const model::Binary &Binary;
  const CFGMap &CFG;
  AssemblyInternalContainer &Output;

  std::unique_ptr<DissassemblyHelper> Helper;
  std::unique_ptr<RawBinaryView> BinaryView;
  model::AssemblyNameBuilder NameBuilder;

public:
  static constexpr llvm::StringRef Name = "ProcessAssembly";
  using Arguments = TypeList<
    PipeRunArgument<const BinariesContainer, "Binaries", "The input binaries">,
    PipeRunArgument<const CFGMap, "CFG", "Per-function CFG data">,
    PipeRunArgument<AssemblyInternalContainer,
                    "Output",
                    "Internal data for disassembly",
                    Access::Write>>;

  ProcessAssembly(const class Model &Model,
                  llvm::StringRef Config,
                  llvm::StringRef DynamicConfig,
                  const BinariesContainer &BinariesContainer,
                  const CFGMap &CFG,
                  AssemblyInternalContainer &Output);
  ~ProcessAssembly();

  static llvm::Error checkPrecondition(const class Model &Model) {
    return RawBinaryView::checkPrecondition(*Model.get().get());
  }

  void runOnFunction(const model::Function &TheFunction);
};

} // namespace piperuns

} // namespace revng::pypeline

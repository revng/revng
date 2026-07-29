#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/Pipes/Containers.h"
#include "revng/Yield/Pipes/YieldControlFlow.h"

class DissassemblyHelper;

namespace revng::pypeline::piperuns {

class ProcessAssembly {
private:
  const model::Binary &Binary;
  const CFGMap &CFG;
  AssemblyInternalContainer &Output;

  std::unique_ptr<DissassemblyHelper> Helper;
  std::unique_ptr<RawBinaryView> BinaryView;
  model::AssemblyNameBuilder NameBuilder;

public:
  static constexpr llvm::StringRef Name = "process-assembly";
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

} // namespace revng::pypeline::piperuns

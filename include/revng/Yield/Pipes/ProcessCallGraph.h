#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/Pipebox/TupleTreeContainer.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"

namespace revng::pypeline {

class CrossRelationsContainer
  : public TupleTreeContainer<yield::crossrelations::CrossRelations,
                              Kinds::Binary> {
public:
  static constexpr llvm::StringRef Name = "CrossRelationsContainer";
  static constexpr llvm::StringRef Compression = "zstd;level=1";
};

namespace piperuns {

class ProcessCallGraph {
private:
  const model::Binary &Binary;
  const CFGMap &Input;
  CrossRelationsContainer &Output;

public:
  static constexpr llvm::StringRef Name = "process-call-graph";
  using Arguments = TypeList<
    PipeRunArgument<const CFGMap, "Input", "CFG map for each function">,
    PipeRunArgument<CrossRelationsContainer,
                    "Output",
                    "Output",
                    Access::Write>>;

  ProcessCallGraph(const Model &Model,
                   llvm::StringRef StaticConfiguration,
                   llvm::StringRef Configuration,
                   const CFGMap &Input,
                   CrossRelationsContainer &Output) :
    Binary(*Model.get().get()), Input(Input), Output(Output){};

  void run();
};

} // namespace piperuns

} // namespace revng::pypeline

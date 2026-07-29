#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/PipeboxCommon/RawContainer.h"
#include "revng/Yield/Pipes/ProcessCallGraph.h"

namespace revng::pypeline {

class CallGraphContainer : public BytesContainer {
public:
  static constexpr llvm::StringRef Name = "CallGraphContainer";
  static constexpr llvm::StringRef MimeType = "image/svg";
  static constexpr llvm::StringRef Compression = "zstd;level=1";
};

namespace piperuns {

class YieldCallGraph {
private:
  const model::Binary &Binary;
  const CrossRelationsContainer &Input;
  CallGraphContainer &Output;

public:
  static constexpr llvm::StringRef Name = "yield-call-graph";
  using Arguments = TypeList<PipeRunArgument<const CrossRelationsContainer,
                                             "Input",
                                             "Binary cross relations">,
                             PipeRunArgument<CallGraphContainer,
                                             "Output",
                                             "SVG of the callgraph",
                                             Access::Write>>;

  YieldCallGraph(const Model &Model,
                 llvm::StringRef StaticConfiguration,
                 llvm::StringRef Configuration,
                 const CrossRelationsContainer &Input,
                 CallGraphContainer &Output) :
    Binary(*Model.get().get()), Input(Input), Output(Output) {}

  void run();
};

} // namespace piperuns

} // namespace revng::pypeline

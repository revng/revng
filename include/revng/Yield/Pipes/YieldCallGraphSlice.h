#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/PTML/Tag.h"
#include "revng/PipeboxCommon/RawContainer.h"
#include "revng/Yield/Pipes/ProcessCallGraph.h"

namespace revng::pypeline {

class CallGraphSliceContainer : public FunctionToBytesContainer {
public:
  static constexpr llvm::StringRef Name = "CallGraphSliceContainer";
  static constexpr llvm::StringRef MimeType = "image/svg";
  static constexpr llvm::StringRef Compression = "zstd;level=1";
};

namespace piperuns {

class YieldCallGraphSlice {
private:
  ptml::MarkupBuilder B;
  const model::Binary &Binary;
  const CrossRelationsContainer &Input;
  CallGraphSliceContainer &Output;

public:
  static constexpr llvm::StringRef Name = "yield-call-graph-slice";
  using Arguments = TypeList<PipeRunArgument<const CrossRelationsContainer,
                                             "Input",
                                             "Binary cross relations">,
                             PipeRunArgument<CallGraphSliceContainer,
                                             "Output",
                                             "per-function SVG of the "
                                             "callgraph",
                                             Access::Write>>;

  YieldCallGraphSlice(const Model &Model,
                      llvm::StringRef StaticConfiguration,
                      llvm::StringRef Configuration,
                      const CrossRelationsContainer &Input,
                      CallGraphSliceContainer &Output) :
    Binary(*Model.get().get()), Input(Input), Output(Output) {}

  void runOnFunction(const model::Function &Function);
};

} // namespace piperuns

} // namespace revng::pypeline

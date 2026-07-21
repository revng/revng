#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/PTML/Tag.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/RawContainer.h"
#include "revng/Yield/Pipes/Containers.h"

namespace revng::pypeline {

class FunctionControlFlowContainer : public FunctionToBytesContainer {
public:
  static constexpr llvm::StringRef Name = "FunctionControlFlowContainer";
  static constexpr llvm::StringRef MimeType = "image/svg";
  static constexpr llvm::StringRef Compression = "zstd;level=1";
};

namespace piperuns {

class YieldCFG {
private:
  ptml::MarkupBuilder B;
  const model::Binary &Binary;
  const AssemblyInternalContainer &Input;
  FunctionControlFlowContainer &Output;

public:
  static constexpr llvm::StringRef Name = "yield-cfg";
  using Arguments = TypeList<PipeRunArgument<const AssemblyInternalContainer,
                                             "Input",
                                             "Internal per-function assembly">,
                             PipeRunArgument<FunctionControlFlowContainer,
                                             "Output",
                                             "per-function CFG with assembly",
                                             Access::Write>>;

  YieldCFG(const Model &Model,
           llvm::StringRef StaticConfiguration,
           llvm::StringRef Configuration,
           const AssemblyInternalContainer &Input,
           FunctionControlFlowContainer &Output) :
    Binary(*Model.get().get()), Input(Input), Output(Output) {}

  void runOnFunction(const model::Function &Function);
};

} // namespace piperuns

} // namespace revng::pypeline

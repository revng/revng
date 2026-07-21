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
#include "revng/Yield/Pipes/ProcessAssembly.h"
#include "revng/Yield/Pipes/YieldControlFlow.h"

namespace revng::pypeline {

class AssemblyContainer : public FunctionToBytesContainer {
public:
  static constexpr llvm::StringRef Name = "AssemblyContainer";
  static constexpr llvm::StringRef MimeType = "text/x.asm+ptml";
  static constexpr llvm::StringRef Compression = "zstd;level=1";
};

namespace piperuns {

class YieldAssembly {
private:
  const model::Binary &Model;
  const AssemblyInternalContainer &Input;
  AssemblyContainer &Output;

  model::CNameBuilder NameBuilder;
  ptml::MarkupBuilder B;

public:
  static constexpr llvm::StringRef Name = "yield-assembly";
  using Arguments = TypeList<PipeRunArgument<const AssemblyInternalContainer,
                                             "Input",
                                             "The internal disassembly data">,
                             PipeRunArgument<AssemblyContainer,
                                             "Output",
                                             "Per-function disassembly",
                                             Access::Write>>;

  YieldAssembly(const class Model &Model,
                llvm::StringRef Config,
                llvm::StringRef DynamicConfig,
                const AssemblyInternalContainer &Input,
                AssemblyContainer &Output) :
    Model(*Model.get().get()),
    Input(Input),
    Output(Output),
    NameBuilder(*Model.get().get()) {}

  void runOnFunction(const model::Function &TheFunction);
};

} // namespace piperuns

} // namespace revng::pypeline

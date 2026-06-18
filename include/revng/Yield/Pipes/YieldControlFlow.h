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
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Yield/Pipes/Containers.h"

namespace revng::pipes {

inline constexpr char FunctionAssemblyYamlMIMEType[] = "text/x.yaml";
inline constexpr char FunctionAssemblyYamlName[] = "function-assembly-internal";
inline constexpr char FunctionAssemblyYamlExtension[] = ".yml";
using FunctionAssemblyStringMap = FunctionStringMap<
  &kinds::FunctionAssemblyInternal,
  FunctionAssemblyYamlName,
  FunctionAssemblyYamlMIMEType,
  FunctionAssemblyYamlExtension>;

inline constexpr char FunctionAssemblyPTMLMIMEType[] = "text/x.asm+ptml+tar+gz";
inline constexpr char FunctionAssemblyPTMLName[] = "function-assembly-ptml";
inline constexpr char FunctionAssemblyPTMLExtension[] = ".asm.tar.gz";

using FunctionAssemblyPTMLStringMap = FunctionStringMap<
  &kinds::FunctionAssemblyPTML,
  FunctionAssemblyPTMLName,
  FunctionAssemblyPTMLMIMEType,
  FunctionAssemblyPTMLExtension>;

inline constexpr char FunctionControlFlowMIMEType[] = "image/svg";
#define NAME "function-control-flow-graph-svg"
inline constexpr char FunctionControlFlowName[] = NAME;
#undef NAME
inline constexpr char FunctionControlFlowExtension[] = ".svg";
using FunctionControlFlowStringMap = FunctionStringMap<
  &kinds::FunctionControlFlowGraphSVG,
  FunctionControlFlowName,
  FunctionControlFlowMIMEType,
  FunctionControlFlowExtension>;

class YieldControlFlow {
public:
  static constexpr auto Name = "yield-cfg";

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(kinds::FunctionAssemblyInternal,
                                     0,
                                     kinds::FunctionControlFlowGraphSVG,
                                     1,
                                     pipeline::InputPreservation::Preserve) };
  }

public:
  void run(pipeline::ExecutionContext &Context,
           const FunctionAssemblyStringMap &Input,
           FunctionControlFlowStringMap &Output);
};

} // namespace revng::pipes

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

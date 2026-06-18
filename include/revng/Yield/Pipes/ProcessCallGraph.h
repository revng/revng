#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/Pipebox/TupleTreeContainer.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringBufferContainer.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Pipes/TupleTreeContainer.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"

namespace revng::pipes {

inline constexpr char CrossRelationsFileMIMEType[] = "text/x.yaml";
inline constexpr char CrossRelationsFileSuffix[] = "";
inline constexpr char CrossRelationsName[] = "binary-cross-relations";

using CrossRelationsFileContainer = pipes::TupleTreeContainer<
  yield::crossrelations::CrossRelations,
  &kinds::BinaryCrossRelations,
  CrossRelationsName,
  CrossRelationsFileMIMEType>;

inline constexpr char CallGraphSVGMIMEType[] = "image/svg";
inline constexpr char CallGraphSVGSuffix[] = ".svg";
inline constexpr char CallGraphSVGName[] = "call-graph-svg";

using CallGraphSVGFileContainer = StringBufferContainer<&kinds::CallGraphSVG,
                                                        CallGraphSVGName,
                                                        CallGraphSVGMIMEType,
                                                        CallGraphSVGSuffix>;

inline constexpr char CallGraphSliceMIMEType[] = "image/svg";
inline constexpr char CallGraphSliceName[] = "call-graph-slice-svg";

using CallGraphSliceSVGStringMap = FunctionStringMap<&kinds::CallGraphSliceSVG,
                                                     CallGraphSliceName,
                                                     CallGraphSliceMIMEType,
                                                     CallGraphSVGSuffix>;

class ProcessCallGraph {
public:
  static constexpr const auto Name = "process-call-graph";

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    return { ContractGroup(kinds::CFG,
                           0,
                           kinds::BinaryCrossRelations,
                           1,
                           pipeline::InputPreservation::Preserve) };
  }

public:
  void run(pipeline::ExecutionContext &Context,
           const CFGMap &CFGMap,
           CrossRelationsFileContainer &OutputFile);
};

} // namespace revng::pipes

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

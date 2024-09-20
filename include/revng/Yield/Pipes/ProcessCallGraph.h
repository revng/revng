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

  llvm::Error checkPrecondition(const pipeline::Context &Context) const {
    return llvm::Error::success();
  }
};

} // namespace revng::pipes

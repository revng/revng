//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Pipes/TupleTreeContainer.h"
#include "revng/TupleTree/TupleTree.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"
#include "revng/Yield/Generated/ForwardDecls.h"
#include "revng/Yield/Pipes/ProcessCallGraph.h"
#include "revng/Yield/Pipes/YieldCallGraph.h"
#include "revng/Yield/Pipes/YieldCallGraphSlice.h"
#include "revng/Yield/SVG.h"

namespace revng::pipes {

void ProcessCallGraph::run(pipeline::ExecutionContext &Context,
                           const CFGMap &CFGMap,
                           CrossRelationsFileContainer &OutputFile) {
  if (Context.getRequestedTargetsFor(OutputFile).empty())
    return;

  // Access the model
  const auto &Model = getModelFromContext(Context);

  // Gather function metadata
  SortedVector<efa::ControlFlowGraph> Metadata;
  for (const auto &[Address, CFGString] : CFGMap) {
    auto MaybeCFG = TupleTree<efa::ControlFlowGraph>::fromString(CFGString);
    Metadata.insert(*llvm::cantFail(std::move(MaybeCFG)));
  }

  // If some functions are missing, do not output anything
  if (Metadata.size() != Model->Functions().size())
    return;

  OutputFile.emplace(Metadata, *Model);

  Context.commitUniqueTarget(OutputFile);
}

llvm::Error YieldCallGraph::run(pipeline::ExecutionContext &Context,
                                const CrossRelationsFileContainer &Relations,
                                CallGraphSVGFileContainer &Output) {
  if (llvm::Error Error = Relations.verify())
    return Error;

  // Access the model
  const auto &Model = revng::getModelFromContext(Context);
  ptml::MarkupBuilder B;

  // Convert the graph to SVG.
  auto Result = yield::svg::callGraph(B, *Relations.get(), *Model);

  // Print the result.
  Output.setContent(std::move(Result));

  Context.commitUniqueTarget(Output);

  return llvm::Error::success();
}

llvm::Error
YieldCallGraphSlice::run(pipeline::ExecutionContext &Context,
                         const CFGMap &CFGMap,
                         const CrossRelationsFileContainer &Relations,
                         CallGraphSliceSVGStringMap &Output) {
  if (llvm::Error Error = Relations.verify())
    return Error;

  // Access the model
  const auto &Model = revng::getModelFromContext(Context);

  // Access the llvm module
  ptml::MarkupBuilder B;

  ControlFlowGraphCache Cache(CFGMap);

  for (const model::Function &Function :
       getFunctionsAndCommit(Context, Output.name())) {
    auto &Metadata = Cache.getControlFlowGraph(Function.Entry());

    // Slice the graph for the current function and convert it to SVG
    auto SlicePoint = pipeline::locationString(revng::ranks::Function,
                                               Metadata.Entry());
    Output.insert_or_assign(Function.Entry(),
                            yield::svg::callGraphSlice(B,
                                                       SlicePoint,
                                                       *Relations.get(),
                                                       *Model));
  }

  return llvm::Error::success();
}

using namespace pipeline;

static RegisterDefaultConstructibleContainer<CrossRelationsFileContainer> X1;
static RegisterDefaultConstructibleContainer<CallGraphSVGFileContainer> X2;
static RegisterDefaultConstructibleContainer<CallGraphSliceSVGStringMap> X3;

static pipeline::RegisterRole Role("binary-cross-relations",
                                   kinds::BinaryCrossRelationsRole);

static pipeline::RegisterPipe<ProcessCallGraph> ProcessPipe;
static pipeline::RegisterPipe<YieldCallGraph> YieldPipe;
static pipeline::RegisterPipe<YieldCallGraphSlice> YieldSlicePipe;

} // end namespace revng::pipes

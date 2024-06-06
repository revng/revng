//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
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

using ptml::PTMLBuilder;

namespace revng::pipes {

void ProcessCallGraph::run(pipeline::ExecutionContext &Context,
                           const CFGMap &CFGMap,
                           CrossRelationsFileContainer &OutputFile) {
  // Access the model
  const auto &Model = getModelFromContext(Context);

  // Gather function metadata
  SortedVector<efa::FunctionMetadata> Metadata;
  for (const auto &[Address, CFGString] : CFGMap)
    Metadata
      .insert(*TupleTree<efa::FunctionMetadata>::deserialize(CFGString)->get());

  // If some functions are missing, do not output anything
  if (Metadata.size() != Model->Functions().size())
    return;

  OutputFile.emplace(Metadata, *Model);
}

void ProcessCallGraph::print(const pipeline::Context &,
                             llvm::raw_ostream &OS,
                             llvm::ArrayRef<std::string>) const {
  OS << "[this is a pure pipe, no command exists for its invocation]\n";
}

void YieldCallGraph::run(pipeline::ExecutionContext &Context,
                         const CrossRelationsFileContainer &Relations,
                         CallGraphSVGFileContainer &Output) {
  // Access the model
  const auto &Model = revng::getModelFromContext(Context);
  PTMLBuilder ThePTMLBuilder;

  // Convert the graph to SVG.
  auto Result = yield::svg::callGraph(ThePTMLBuilder, *Relations.get(), *Model);

  // Print the result.
  Output.setContent(std::move(Result));
}

void YieldCallGraph::print(const pipeline::Context &,
                           llvm::raw_ostream &OS,
                           llvm::ArrayRef<std::string>) const {
  OS << "[this is a pure pipe, no command exists for its invocation]\n";
}

void YieldCallGraphSlice::run(pipeline::ExecutionContext &Context,
                              const CFGMap &CFGMap,
                              const CrossRelationsFileContainer &Relations,
                              CallGraphSliceSVGStringMap &Output) {
  // Access the model
  const auto &Model = revng::getModelFromContext(Context);

  // Access the llvm module
  PTMLBuilder ThePTMLBuilder;

  FunctionMetadataCache Cache(CFGMap);
  for (const auto &[Key, _] : CFGMap) {
    MetaAddress Address = std::get<0>(Key);
    auto &Metadata = Cache.getFunctionMetadata(Address);
    revng_assert(llvm::is_contained(Model->Functions(), Metadata.Entry()));

    // Slice the graph for the current function and convert it to SVG
    auto SlicePoint = pipeline::serializedLocation(revng::ranks::Function,
                                                   Metadata.Entry());
    Output.insert_or_assign(Address,
                            yield::svg::callGraphSlice(ThePTMLBuilder,
                                                       SlicePoint,
                                                       *Relations.get(),
                                                       *Model));
  }
}

void YieldCallGraphSlice::print(const pipeline::Context &,
                                llvm::raw_ostream &OS,
                                llvm::ArrayRef<std::string>) const {
  OS << "[this is a pure pipe, no command exists for its invocation]\n";
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

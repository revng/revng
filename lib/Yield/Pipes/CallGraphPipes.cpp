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
#include "revng/Pipes/FunctionStringMap.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/TupleTreeContainer.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"
#include "revng/Yield/Generated/ForwardDecls.h"
#include "revng/Yield/Pipes/ProcessCallGraph.h"
#include "revng/Yield/Pipes/YieldCallGraph.h"
#include "revng/Yield/Pipes/YieldCallGraphSlice.h"
#include "revng/Yield/SVG.h"

namespace revng::pipes {

void ProcessCallGraph::run(pipeline::Context &Context,
                           const pipeline::LLVMContainer &TargetList,
                           CrossRelationsFileContainer &OutputFile) {
  // Access the model
  const auto &Model = getModelFromContext(Context);

  // Access the llvm module
  const llvm::Module &Module = TargetList.getModule();

  // Gather function metadata
  SortedVector<efa::FunctionMetadata> Metadata;
  for (const auto &LLVMFunction : FunctionTags::Isolated.functions(&Module))
    Metadata.insert(*::detail::extractFunctionMetadata(&LLVMFunction));
  revng_assert(Metadata.size() == Model->Functions().size());

  OutputFile.emplace(Metadata, *Model);
}

void ProcessCallGraph::print(const pipeline::Context &,
                             llvm::raw_ostream &OS,
                             llvm::ArrayRef<std::string>) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

void YieldCallGraph::run(pipeline::Context &Context,
                         const CrossRelationsFileContainer &Relations,
                         CallGraphSVGFileContainer &Output) {
  // Access the model
  const auto &Model = revng::getModelFromContext(Context);

  // Convert the graph to SVG.
  auto Result = yield::svg::callGraph(*Relations.get(), *Model);

  // Print the result.
  Output.setContent(std::move(Result));
}

void YieldCallGraph::print(const pipeline::Context &,
                           llvm::raw_ostream &OS,
                           llvm::ArrayRef<std::string>) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

void YieldCallGraphSlice::run(pipeline::Context &Context,
                              const pipeline::LLVMContainer &TargetList,
                              const CrossRelationsFileContainer &Relations,
                              CallGraphSliceSVGStringMap &Output) {
  // Access the model
  const auto &Model = revng::getModelFromContext(Context);

  // Access the llvm module
  const llvm::Module &Module = TargetList.getModule();
  FunctionMetadataCache Cache;
  for (const auto &LLVMFunction : FunctionTags::Isolated.functions(&Module)) {
    auto &Metadata = Cache.getFunctionMetadata(&LLVMFunction);
    auto ModelFunctionIterator = Model->Functions().find(Metadata.Entry());
    revng_assert(ModelFunctionIterator != Model->Functions().end());

    // Slice the graph for the current function and convert it to SVG
    Output.insert_or_assign(Metadata.Entry(),
                            yield::svg::callGraphSlice(Metadata.Entry(),
                                                       *Relations.get(),
                                                       *Model));
  }
}

void YieldCallGraphSlice::print(const pipeline::Context &,
                                llvm::raw_ostream &OS,
                                llvm::ArrayRef<std::string>) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}
using namespace pipeline;

static RegisterDefaultConstructibleContainer<CrossRelationsFileContainer> X1;
static RegisterDefaultConstructibleContainer<CallGraphSVGFileContainer> X2;
static RegisterFunctionStringMap<CallGraphSliceSVGStringMap> X3;

static pipeline::RegisterRole
  Role("BinaryCrossRelations", kinds::BinaryCrossRelationsRole);

static pipeline::RegisterPipe<ProcessCallGraph> ProcessPipe;
static pipeline::RegisterPipe<YieldCallGraph> YieldPipe;
static pipeline::RegisterPipe<YieldCallGraphSlice> YieldSlicePipe;

} // end namespace revng::pipes

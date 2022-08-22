//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Yield/CrossRelations.h"
#include "revng/Yield/Pipes/ProcessCallGraphPipe.h"
#include "revng/Yield/Pipes/YieldCallGraphPipe.h"
#include "revng/Yield/Pipes/YieldCallGraphSlicePipe.h"
#include "revng/Yield/SVG.h"

namespace revng::pipes {

void ProcessCallGraph::run(pipeline::Context &Context,
                           const pipeline::LLVMContainer &TargetList,
                           FileContainer &OutputFile) {
  // Access the model
  const auto &Model = getModelFromContext(Context);

  // Access the llvm module
  const llvm::Module &Module = TargetList.getModule();

  // Gather function metadata
  SortedVector<efa::FunctionMetadata> Metadata;
  for (const auto &LLVMFunction : FunctionTags::Isolated.functions(&Module))
    Metadata.insert(*extractFunctionMetadata(&LLVMFunction));
  revng_assert(Metadata.size() == Model->Functions.size());

  // Gather the relations
  yield::CrossRelations Relations(Metadata, *Model);

  // Serialize the output
  std::error_code ErrorCode;
  llvm::raw_fd_ostream Stream(OutputFile.getOrCreatePath(), ErrorCode);
  if (ErrorCode)
    revng_abort(ErrorCode.message().c_str());

  llvm::yaml::Output YamlStream(Stream);
  YamlStream << Relations;

  Stream.flush();
  if ((ErrorCode = Stream.error()))
    revng_abort(ErrorCode.message().c_str());
}

void ProcessCallGraph::print(const pipeline::Context &,
                             llvm::raw_ostream &OS,
                             llvm::ArrayRef<std::string>) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

void YieldCallGraph::run(pipeline::Context &Context,
                         const FileContainer &Input,
                         FileContainer &Output) {
  // Access the model
  const auto &Model = revng::getModelFromContext(Context);

  // Open the input file.
  auto MaybeInputPath = Input.path();
  revng_assert(MaybeInputPath.has_value());
  auto MaybeBuffer = llvm::MemoryBuffer::getFile(MaybeInputPath.value());
  revng_assert(MaybeBuffer);
  llvm::yaml::Input YAMLInput(**MaybeBuffer);

  // Deserialize the graph data.
  yield::CrossRelations Relations;
  YAMLInput >> Relations;

  // Convert the graph to SVG.
  auto Result = yield::svg::callGraph(Relations, *Model);

  // Open the output file.
  std::error_code ErrorCode;
  llvm::raw_fd_ostream Stream(Output.getOrCreatePath(), ErrorCode);
  if (ErrorCode)
    revng_abort(ErrorCode.message().c_str());

  // Print the result.
  Stream << Result;
  Stream.flush();
  if ((ErrorCode = Stream.error()))
    revng_abort(ErrorCode.message().c_str());
}

void YieldCallGraph::print(const pipeline::Context &,
                           llvm::raw_ostream &OS,
                           llvm::ArrayRef<std::string>) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

void YieldCallGraphSlice::run(pipeline::Context &Context,
                              const pipeline::LLVMContainer &TargetList,
                              const FileContainer &Input,
                              FunctionStringMap &Output) {
  // Access the model
  const auto &Model = revng::getModelFromContext(Context);

  // Open the input file.
  auto MaybeInputPath = Input.path();
  revng_assert(MaybeInputPath.has_value());
  auto MaybeBuffer = llvm::MemoryBuffer::getFile(MaybeInputPath.value());
  revng_assert(MaybeBuffer);
  llvm::yaml::Input YAMLInput(**MaybeBuffer);

  // Deserialize the graph data.
  yield::CrossRelations Relations;
  YAMLInput >> Relations;

  // Access the llvm module
  const llvm::Module &Module = TargetList.getModule();
  for (const auto &LLVMFunction : FunctionTags::Isolated.functions(&Module)) {
    auto Metadata = extractFunctionMetadata(&LLVMFunction);
    auto ModelFunctionIterator = Model->Functions.find(Metadata->Entry);
    revng_assert(ModelFunctionIterator != Model->Functions.end());

    // Slice the graph for the current function and convert it to SVG
    Output.insert_or_assign(Metadata->Entry,
                            yield::svg::callGraphSlice(Metadata->Entry,
                                                       Relations,
                                                       *Model));
  }
}

void YieldCallGraphSlice::print(const pipeline::Context &,
                                llvm::raw_ostream &OS,
                                llvm::ArrayRef<std::string>) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

static pipeline::RegisterContainerFactory
  InternalContainer("BinaryCrossRelations",
                    makeFileContainerFactory(kinds::BinaryCrossRelations,
                                             "application/"
                                             "x.yaml.cross-relations"));
static pipeline::RegisterContainerFactory
  FullGraphContainer("CallGraphSVG",
                     makeFileContainerFactory(kinds::CallGraphSVG,
                                              "application/"
                                              "x.yaml.call-graph.svg-body"));
static revng::pipes::RegisterFunctionStringMap
  GraphSliceContainer("CallGraphSliceSVG",
                      "application/x.yaml.call-graph-slice.svg-body",
                      kinds::CallGraphSliceSVG);

static pipeline::RegisterRole
  Role("BinaryCrossRelations", kinds::BinaryCrossRelationsRole);

static pipeline::RegisterPipe<ProcessCallGraph> ProcessPipe;
static pipeline::RegisterPipe<YieldCallGraph> YieldPipe;
static pipeline::RegisterPipe<YieldCallGraphSlice> YieldSlicePipe;

} // end namespace revng::pipes

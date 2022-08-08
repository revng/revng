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

namespace revng::pipes {

void ProcessCallGraphPipe::run(pipeline::Context &Context,
                               const pipeline::LLVMContainer &TargetList,
                               FileContainer &OutputFile) {
  // Access the model
  const auto &Model = revng::pipes::getModelFromContext(Context);

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

void ProcessCallGraphPipe::print(const pipeline::Context &,
                                 llvm::raw_ostream &OS,
                                 llvm::ArrayRef<std::string>) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

std::array<pipeline::ContractGroup, 1>
ProcessCallGraphPipe::getContract() const {
  return { pipeline::ContractGroup(IsolatedRoot,
                                   pipeline::Exactness::Exact,
                                   0,
                                   BinaryCrossRelations,
                                   1,
                                   pipeline::InputPreservation::Preserve) };
}

static pipeline::RegisterContainerFactory
  InternalContainer("BinaryCrossRelations",
                    makeFileContainerFactory(BinaryCrossRelations,
                                             "application/"
                                             "x.yaml.cross-relations"));

static pipeline::RegisterRole
  Role("BinaryCrossRelations", BinaryCrossRelationsRole);

static pipeline::RegisterPipe<ProcessCallGraphPipe> ProcessPipe;

} // end namespace revng::pipes

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Lift/LoadBinaryPass.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/Yield/Assembly.h"
#include "revng/Yield/Assembly/Assembly.h"
#include "revng/Yield/Assembly/DisassemblyHelper.h"
#include "revng/Yield/HTML.h"

namespace revng::pipes {

void YieldAssemblyPipe::run(pipeline::Context &Context,
                            const FileContainer &SourceBinary,
                            const pipeline::LLVMContainer &TargetList,
                            FunctionStringMap &Output) {
  // Access the model
  const auto &Model = revng::pipes::getModelFromContext(Context);

  // Access the binary
  revng_assert(SourceBinary.path().has_value());
  auto MaybeBinary = loadBinary(*Model, *SourceBinary.path());
  revng_assert(MaybeBinary);
  const RawBinaryView &BinaryView = MaybeBinary->first;

  // Access the llvm module
  const llvm::Module &Module = TargetList.getModule();

  // Define the helper object to store the disassembly pipeline.
  // This allows it to only be created once.
  DissassemblyHelper Helper;

  for (const auto &LLVMFunction : FunctionTags::Isolated.functions(&Module)) {
    auto Metadata = extractFunctionMetadata(&LLVMFunction);
    auto ModelFunctionIterator = Model->Functions.find(Metadata->Entry);
    revng_assert(ModelFunctionIterator != Model->Functions.end());
    const auto &Function = *ModelFunctionIterator;

    auto Disassembled = Helper.disassemble(Function, *Metadata, BinaryView);

    auto HTML = yield::html::assembly(Disassembled, *Metadata, *Model);
    Output.insert_or_assign(Function.Entry, std::move(HTML));
  }
}

void YieldAssemblyPipe::print(const pipeline::Context &,
                              llvm::raw_ostream &OS,
                              llvm::ArrayRef<std::string> Files) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

std::array<pipeline::ContractGroup, 1> YieldAssemblyPipe::getContract() const {
  pipeline::Contract BinaryContract(Binary,
                                    pipeline::Exactness::Exact,
                                    0,
                                    pipeline::InputPreservation::Preserve);

  pipeline::Contract FunctionContract(Isolated,
                                      pipeline::Exactness::Exact,
                                      1,
                                      FunctionAssemblyHTML,
                                      2,
                                      pipeline::InputPreservation::Preserve);

  return { pipeline::ContractGroup{ std::move(BinaryContract),
                                    std::move(FunctionContract) } };
}

} // end namespace revng::pipes

using revng::pipes::FunctionAssemblyHTML;
using revng::pipes::FunctionStringMap;
static revng::pipes::RegisterFunctionStringMap
  AssemblyContainer("FunctionAssemblyHTML",
                    "application/"
                    "x.yaml.function-assembly"
                    ".html-body",
                    FunctionAssemblyHTML);

static pipeline::RegisterPipe<revng::pipes::YieldAssemblyPipe> AssemblyPipe;

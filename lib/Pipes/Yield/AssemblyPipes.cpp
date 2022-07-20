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
#include "revng/Pipes/Yield/ProcessAssemblyPipe.h"
#include "revng/Pipes/Yield/YieldAssemblyPipe.h"
#include "revng/Yield/Assembly/DisassemblyHelper.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/PTML.h"

namespace revng::pipes {

void ProcessAssemblyPipe::run(pipeline::Context &Context,
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

    const auto &Func = *ModelFunctionIterator;
    auto Disassembled = Helper.disassemble(Func, *Metadata, BinaryView, *Model);
    Output.insert_or_assign(Func.Entry, serializeToString(Disassembled));
  }
}

void ProcessAssemblyPipe::print(const pipeline::Context &,
                                llvm::raw_ostream &OS,
                                llvm::ArrayRef<std::string> Files) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

std::array<pipeline::ContractGroup, 1>
ProcessAssemblyPipe::getContract() const {
  pipeline::Contract BinaryContract(Binary,
                                    pipeline::Exactness::Exact,
                                    0,
                                    pipeline::InputPreservation::Preserve);

  pipeline::Contract FunctionContract(Isolated,
                                      pipeline::Exactness::Exact,
                                      1,
                                      FunctionAssemblyInternal,
                                      2,
                                      pipeline::InputPreservation::Preserve);

  return { pipeline::ContractGroup{ std::move(BinaryContract),
                                    std::move(FunctionContract) } };
}

void YieldAssemblyPipe::run(pipeline::Context &Context,
                            const FunctionStringMap &Input,
                            FunctionStringMap &Output) {
  // Access the model
  const auto &Model = revng::pipes::getModelFromContext(Context);

  for (auto [Address, S] : Input) {
    auto MaybeFunction = TupleTree<yield::Function>::deserialize(S);
    revng_assert(MaybeFunction && MaybeFunction->verify());
    revng_assert((*MaybeFunction)->Entry == Address);

    Output.insert_or_assign((*MaybeFunction)->Entry,
                            yield::ptml::functionAssembly(**MaybeFunction,
                                                          *Model));
  }
}

void YieldAssemblyPipe::print(const pipeline::Context &,
                              llvm::raw_ostream &OS,
                              llvm::ArrayRef<std::string> Files) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

std::array<pipeline::ContractGroup, 1> YieldAssemblyPipe::getContract() const {
  return { pipeline::ContractGroup(FunctionAssemblyInternal,
                                   pipeline::Exactness::Exact,
                                   0,
                                   FunctionAssemblyPTML,
                                   1,
                                   pipeline::InputPreservation::Preserve) };
}

} // end namespace revng::pipes

static revng::pipes::RegisterFunctionStringMap
  InternalContainer("FunctionAssemblyInternal",
                    "application/x.yaml.function-assembly.internal",
                    revng::pipes::FunctionAssemblyInternal);
static revng::pipes::RegisterFunctionStringMap
  PTMLContainer("FunctionAssemblyPTML",
                "application/x.yaml.function-assembly.ptml-body",
                revng::pipes::FunctionAssemblyPTML);

static pipeline::RegisterPipe<revng::pipes::ProcessAssemblyPipe> ProcessPipe;
static pipeline::RegisterPipe<revng::pipes::YieldAssemblyPipe> YieldPipe;

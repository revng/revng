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
#include "revng/Yield/Assembly/DisassemblyHelper.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/PTML.h"
#include "revng/Yield/Pipes/ProcessAssemblyPipe.h"
#include "revng/Yield/Pipes/YieldAssemblyPipe.h"

namespace revng::pipes {

void ProcessAssembly::run(pipeline::Context &Context,
                          const FileContainer &SourceBinary,
                          const pipeline::LLVMContainer &TargetList,
                          FunctionStringMap &Output) {
  if (not SourceBinary.exists())
    return;

  // Access the model
  const auto &Model = getModelFromContext(Context);

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

void ProcessAssembly::print(const pipeline::Context &,
                            llvm::raw_ostream &OS,
                            llvm::ArrayRef<std::string> Files) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

void YieldAssembly::run(pipeline::Context &Context,
                        const FunctionStringMap &Input,
                        FunctionStringMap &Output) {
  // Access the model
  const auto &Model = getModelFromContext(Context);

  for (auto [Address, S] : Input) {
    auto MaybeFunction = TupleTree<yield::Function>::deserialize(S);
    revng_assert(MaybeFunction && MaybeFunction->verify());
    revng_assert((*MaybeFunction)->Entry == Address);

    Output.insert_or_assign((*MaybeFunction)->Entry,
                            yield::ptml::functionAssembly(**MaybeFunction,
                                                          *Model));
  }
}

void YieldAssembly::print(const pipeline::Context &,
                          llvm::raw_ostream &OS,
                          llvm::ArrayRef<std::string> Files) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

} // end namespace revng::pipes

static revng::pipes::RegisterFunctionStringMap
  InternalContainer("FunctionAssemblyInternal",
                    "application/x.yaml.function-assembly.internal",
                    revng::kinds::FunctionAssemblyInternal);
static revng::pipes::RegisterFunctionStringMap
  PTMLContainer("FunctionAssemblyPTML",
                "application/x.yaml.function-assembly.ptml-body",
                revng::kinds::FunctionAssemblyPTML);

static pipeline::RegisterPipe<revng::pipes::ProcessAssembly> ProcessPipe;
static pipeline::RegisterPipe<revng::pipes::YieldAssembly> YieldPipe;

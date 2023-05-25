//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Lift/LoadBinaryPass.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Yield/Assembly/DisassemblyHelper.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/PTML.h"
#include "revng/Yield/Pipes/ProcessAssembly.h"
#include "revng/Yield/Pipes/YieldAssembly.h"

using ptml::PTMLBuilder;

namespace revng::pipes {

void ProcessAssembly::run(pipeline::Context &Context,
                          const BinaryFileContainer &SourceBinary,
                          const pipeline::LLVMContainer &TargetList,
                          FunctionAssemblyStringMap &Output) {
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

  FunctionMetadataCache Cache;
  for (const auto &LLVMFunction : FunctionTags::Isolated.functions(&Module)) {
    const auto &Metadata = Cache.getFunctionMetadata(&LLVMFunction);
    auto ModelFunctionIterator = Model->Functions().find(Metadata.Entry());
    revng_assert(ModelFunctionIterator != Model->Functions().end());

    const auto &Func = *ModelFunctionIterator;
    auto Disassembled = Helper.disassemble(Func, Metadata, BinaryView, *Model);
    Output.insert_or_assign(Func.Entry(), serializeToString(Disassembled));
  }
}

void ProcessAssembly::print(const pipeline::Context &,
                            llvm::raw_ostream &OS,
                            llvm::ArrayRef<std::string> Files) const {
  OS << "[this is a pure pipe, no command exists for its invocation]\n";
}

void YieldAssembly::run(pipeline::Context &Context,
                        const FunctionAssemblyStringMap &Input,
                        FunctionAssemblyPTMLStringMap &Output) {
  // Access the model
  const auto &Model = getModelFromContext(Context);

  PTMLBuilder ThePTMLBuilder;
  for (auto [Address, S] : Input) {
    auto MaybeFunction = TupleTree<yield::Function>::deserialize(S);
    revng_assert(MaybeFunction && MaybeFunction->verify());
    revng_assert((*MaybeFunction)->Entry() == Address);

    Output.insert_or_assign((*MaybeFunction)->Entry(),
                            yield::ptml::functionAssembly(ThePTMLBuilder,
                                                          **MaybeFunction,
                                                          *Model));
  }
}

void YieldAssembly::print(const pipeline::Context &,
                          llvm::raw_ostream &OS,
                          llvm::ArrayRef<std::string> Files) const {
  OS << "[this is a pure pipe, no command exists for its invocation]\n";
}

} // end namespace revng::pipes

using namespace revng::pipes;
static RegisterFunctionStringMap<FunctionAssemblyStringMap> X1;
static RegisterFunctionStringMap<FunctionAssemblyPTMLStringMap> X2;

static pipeline::RegisterPipe<revng::pipes::ProcessAssembly> ProcessPipe;
static pipeline::RegisterPipe<revng::pipes::YieldAssembly> YieldPipe;

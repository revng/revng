//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Lift/LoadBinaryPass.h"
#include "revng/Model/Binary.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/Doxygen.h"
#include "revng/PTML/Tag.h"
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

void ProcessAssembly::run(pipeline::ExecutionContext &Context,
                          const BinaryFileContainer &SourceBinary,
                          const CFGMap &CFGMap,
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

  // Define the helper object to store the disassembly pipeline.
  // This allows it to only be created once.
  DissassemblyHelper Helper;

  FunctionMetadataCache Cache(CFGMap);
  for (const auto &[Key, _] : CFGMap) {
    MetaAddress Address = std::get<0>(Key);
    const auto &Metadata = Cache.getFunctionMetadata(Address);
    auto ModelFunctionIterator = Model->Functions().find(Address);
    revng_assert(ModelFunctionIterator != Model->Functions().end());

    const auto &Func = *ModelFunctionIterator;
    auto Disassembled = Helper.disassemble(Func, Metadata, BinaryView, *Model);
    Output.insert_or_assign(Func.Entry(), serializeToString(Disassembled));
  }
}

void YieldAssembly::run(pipeline::ExecutionContext &Context,
                        const FunctionAssemblyStringMap &Input,
                        FunctionAssemblyPTMLStringMap &Output) {
  // Access the model
  const auto &Model = getModelFromContext(Context);

  PTMLBuilder ThePTMLBuilder;
  for (auto [Address, S] : Input) {
    auto MaybeFunction = TupleTree<yield::Function>::deserialize(S);
    revng_assert(MaybeFunction && MaybeFunction->verify());
    revng_assert((*MaybeFunction)->Entry() == std::get<0>(Address));

    const model::Function &ModelFunction = Model->Functions()
                                             .at(std::get<0>(Address));
    const model::Architecture::Values A = Model->Architecture();
    auto CommentIndicator = model::Architecture::getAssemblyCommentIndicator(A);
    std::string R = ptml::functionComment(ThePTMLBuilder,
                                          ModelFunction,
                                          *Model,
                                          CommentIndicator,
                                          0,
                                          80);
    R += yield::ptml::functionAssembly(ThePTMLBuilder, **MaybeFunction, *Model);
    R = ThePTMLBuilder.getTag(ptml::tags::Div, std::move(R)).serialize();
    Output.insert_or_assign((*MaybeFunction)->Entry(), std::move(R));
  }
}

} // end namespace revng::pipes

using namespace revng::pipes;
using namespace pipeline;
static RegisterDefaultConstructibleContainer<FunctionAssemblyStringMap> X1;
static RegisterDefaultConstructibleContainer<FunctionAssemblyPTMLStringMap> X2;

static pipeline::RegisterPipe<revng::pipes::ProcessAssembly> ProcessPipe;
static pipeline::RegisterPipe<revng::pipes::YieldAssembly> YieldPipe;

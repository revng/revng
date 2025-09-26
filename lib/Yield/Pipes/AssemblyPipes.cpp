//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Lift/LoadBinaryPass.h"
#include "revng/Model/Binary.h"
#include "revng/Model/NameBuilder.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/Doxygen.h"
#include "revng/PTML/Tag.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/Yield/Assembly/DisassemblyHelper.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/PTML.h"
#include "revng/Yield/Pipes/ProcessAssembly.h"
#include "revng/Yield/Pipes/YieldAssembly.h"

namespace revng::pipes {

void ProcessAssembly::run(pipeline::ExecutionContext &Context,
                          const BinaryFileContainer &SourceBinary,
                          const CFGMap &CFGMap,
                          FunctionAssemblyStringMap &Output) {
  if (not SourceBinary.exists())
    return;

  // Access the model
  const auto &Model = getModelFromContext(Context);
  model::AssemblyNameBuilder NameBuilder = *Model;

  // Access the binary
  revng_assert(SourceBinary.path().has_value());
  auto MaybeBinary = cantFail(loadBinary(*Model, *SourceBinary.path()));
  const RawBinaryView &BinaryView = MaybeBinary.first;

  // Define the helper object to store the disassembly pipeline.
  // This allows it to only be created once.
  DissassemblyHelper Helper;

  ControlFlowGraphCache Cache(CFGMap);

  for (const model::Function &Function :
       getFunctionsAndCommit(Context, Output.name())) {

    const auto &Metadata = Cache.getControlFlowGraph(Function.Entry());

    auto Disassembled = Helper.disassemble(Function,
                                           Metadata,
                                           BinaryView,
                                           *Model,
                                           NameBuilder);
    revng_assert(Disassembled.verify());
    Output.insert_or_assign(Function.Entry(), toString(Disassembled));
  }
}

void YieldAssembly::run(pipeline::ExecutionContext &Context,
                        const FunctionAssemblyStringMap &Input,
                        FunctionAssemblyPTMLStringMap &Output) {
  // Access the model
  const auto &Model = getModelFromContext(Context);
  model::CNameBuilder NameBuilder = *Model;

  ptml::MarkupBuilder B;
  for (const model::Function &Function :
       getFunctionsAndCommit(Context, Output.name())) {
    MetaAddress Address = Function.Entry();
    llvm::StringRef YamlText = Input.at(Address);
    auto MaybeFunction = TupleTree<yield::Function>::fromString(YamlText);

    revng_assert(MaybeFunction && MaybeFunction->verify());
    revng_assert((*MaybeFunction)->verify());
    revng_assert((*MaybeFunction)->Entry() == Address);

    const model::Architecture::Values A = Model->Architecture();
    auto CommentIndicator = model::Architecture::getAssemblyCommentIndicator(A);

    const model::Configuration &Configuration = Model->Configuration();
    uint64_t LineWidth = Configuration.CommentLineWidth();

    std::string R = ptml::functionComment(B,
                                          Function,
                                          *Model,
                                          CommentIndicator,
                                          0,
                                          LineWidth,
                                          NameBuilder);
    R += yield::ptml::functionAssembly(B, **MaybeFunction, *Model);
    R = B.getTag(ptml::tags::Div, std::move(R)).toString();
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

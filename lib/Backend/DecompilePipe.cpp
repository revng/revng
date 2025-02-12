//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Backend/DecompileFunction.h"
#include "revng/Backend/DecompilePipe.h"
#include "revng/HeadersGeneration/Options.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/StringMap.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"

namespace revng::pipes {

using namespace pipeline;
static RegisterDefaultConstructibleContainer<DecompileStringMap> Reg;

void Decompile::run(pipeline::ExecutionContext &EC,
                    pipeline::LLVMContainer &IRContainer,
                    const revng::pipes::CFGMap &CFGMap,
                    DecompileStringMap &DecompiledFunctions) {

  llvm::Module &Module = IRContainer.getModule();
  const model::Binary &Model = *getModelFromContext(EC);
  ControlFlowGraphCache Cache(CFGMap);

  namespace options = revng::options;
  ptml::CTypeBuilder
    B(llvm::nulls(),
      Model,
      /* EnableTaglessMode = */ false,
      { .EnableTypeInlining = not options::DisableTypeInlining,
        .EnableStackFrameInlining = not options::DisableStackFrameInlining });
  B.collectInlinableTypes();

  for (const model::Function &Function :
       getFunctionsAndCommit(EC, DecompiledFunctions.name())) {
    auto *F = Module.getFunction(B.NameBuilder.llvmName(Function));
    std::string CCode = decompile(Cache, *F, Model, B);
    DecompiledFunctions.insert_or_assign(Function.Entry(), std::move(CCode));
  }
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::Decompile> Y;

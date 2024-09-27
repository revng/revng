//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/StringMap.h"

#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/Backend/DecompilePipe.h"
#include "revng-c/Pipes/Kinds.h"
#include "revng-c/TypeNames/ModelToPTMLTypeHelpers.h"

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

  // Get all Stack types and all the inlinable types reachable from it,
  // since we want to emit forward declarations for all of them.

  // TODO: use TypeInlineHelper(Model).findTypesToInlineInStacks().
  //       We have temporarily disabled stack-types inlining due to the fact
  //       that type inlining is broken on rare cases involving recursive types
  //       (do to the fact that it uses a different logic than ModelToHeader).
  //       For this reason this is always the empty set for now. When type
  //       inlining will be fixed this can be re-enabled.
  // TODO: once we re-enable this, we need to do disable model tracking and
  //       manually implement invalidation tracking in the appropriate pipes.
  const TypeInlineHelper::StackTypesMap StackTypes;

  for (const model::Function &Function :
       getFunctionsAndCommit(EC, DecompiledFunctions.name())) {
    llvm::Function *F = Module.getFunction(getLLVMFunctionName(Function));
    std::string CCode = decompile(Cache, *F, Model, StackTypes, false);
    DecompiledFunctions.insert_or_assign(Function.Entry(), std::move(CCode));
  }
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::Decompile> Y;

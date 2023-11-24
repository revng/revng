//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/FunctionStringMap.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"

#include "revng-c/Backend/CDecompilationPipe.h"
#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

static RegisterFunctionStringMap<DecompiledCCodeInYAMLStringMap> Reg;

void CDecompilation::run(const pipeline::ExecutionContext &Ctx,
                         pipeline::LLVMContainer &IRContainer,
                         DecompiledCCodeInYAMLStringMap &DecompiledFunctions) {

  llvm::Module &Module = IRContainer.getModule();
  const model::Binary &Model = *getModelFromContext(Ctx);
  FunctionMetadataCache Cache;
  decompile(Cache, Module, Model, DecompiledFunctions);
}

void CDecompilation::print(const pipeline::Context &Ctx,
                           llvm::raw_ostream &OS,
                           llvm::ArrayRef<std::string> Names) const {
  OS << *revng::ResourceFinder.findFile("bin/revng");
  OS << " decompile -m model.yml -i " << Names[0] << " -o " << Names[1];
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::CDecompilation> Y;

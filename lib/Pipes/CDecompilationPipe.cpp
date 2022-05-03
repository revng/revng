//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/FunctionStringMap.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"

#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/Backend/VariableScopeAnalysisPass.h"
#include "revng-c/BeautifyGHAST/BeautifyGHAST.h"
#include "revng-c/Pipes/CDecompilationPipe.h"
#include "revng-c/Pipes/Kinds.h"
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/RestructureCFG.h"

namespace revng::pipes {

static RegisterFunctionStringMap DecompiledYAML("DecompiledCCode",
                                                "application/"
                                                "x.yaml.c.decompiled",
                                                DecompiledToC);

void CDecompilationPipe::run(const pipeline::Context &Ctx,
                             pipeline::LLVMContainer &IRContainer,
                             FileContainer &ModelHeaderFile,
                             FileContainer &HelpersHeaderFile,
                             FunctionStringMap &DecompiledFunctions) {

  llvm::Module &Module = IRContainer.getModule();
  const model::Binary &Model = *getModelFromContext(Ctx);

  for (llvm::Function &F : FunctionTags::Isolated.functions(&Module)) {

    // TODO: this will eventually become a GHASTContainer for revng pipeline
    ASTTree GHAST;

    // Generate the GHAST and beautify it.
    {
      restructureCFG(F, GHAST);
      // TODO: beautification should be optional, but at the moment it's not
      // truly so (if disabled, things crash). We should strive to make it
      // optional for real.
      beautifyAST(F, GHAST);
    }

    // Generated C code for F
    std::string CCode;
    {
      auto TopScopeVariables = collectLocalVariables(F);
      auto NeedsLoopStateVar = hasLoopDispatchers(GHAST);
      llvm::raw_string_ostream DecompiledStream(CCode);
      decompileFunction(F,
                        GHAST,
                        Model,
                        DecompiledStream,
                        TopScopeVariables,
                        NeedsLoopStateVar,
                        ModelHeaderFile.getOrCreatePath(),
                        HelpersHeaderFile.getOrCreatePath());
      DecompiledStream.flush();
    }

    // Push the C code into
    MetaAddress Key = getMetaAddressMetadata(&F, "revng.function.entry");
    DecompiledFunctions.insert_or_assign(Key, std::move(CCode));
  }
}

void CDecompilationPipe::print(const pipeline::Context &Ctx,
                               llvm::raw_ostream &OS,
                               llvm::ArrayRef<std::string> Names) const {
  OS << *revng::ResourceFinder.findFile("bin/revng");
  OS << " opt " << Names[0]
     << " -restructure-cfg -beautify-ghast -collect-local-vars -c-backend";
  OS << " --model-header=" << Names[1];
  OS << " --helpers-header=" << Names[2];
  OS << " --decompiled-functions=" << Names[3] << " -o /dev/null";
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::CDecompilationPipe> Y;

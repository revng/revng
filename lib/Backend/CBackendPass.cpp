//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"

#include "revng-c/Backend/CBackendPass.h"
#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/Backend/VariableScopeAnalysisPass.h"
#include "revng-c/RestructureCFGPass/RestructureCFG.h"
#include "revng-c/Support/FunctionFileHelpers.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

using namespace llvm;

using llvm::cl::NumOccurrencesFlag;

static cl::opt<std::string> DecompiledDir("c-decompiled-dir",
                                          cl::desc("decompiled C code dir"),
                                          cl::value_desc("c-decompiled-dir"),
                                          cl::cat(MainCategory),
                                          NumOccurrencesFlag::Optional);

char BackendPass::ID = 0;

using Register = RegisterPass<BackendPass>;
static Register X("c-backend", "Decompilation Backend Pass", false, false);

BackendPass::BackendPass(std::unique_ptr<llvm::raw_ostream> Out) :
  llvm::FunctionPass{ ID }, Out{ std::move(Out) } {
}

BackendPass::BackendPass() : BackendPass(nullptr) {
}

void BackendPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<RestructureCFG>();
  AU.addRequired<VariableScopeAnalysisPass>();
  AU.setPreservesAll();
}

bool BackendPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  // If the `-single-decompilation` option was passed from command line, skip
  // decompilation for all the functions that are not the selected one.
  if (not TargetFunction.empty())
    if (not F.getName().equals(TargetFunction.c_str()))
      return false;

  // If the -c-decompiled-dir flag was passed, the decompiled function needs to
  // be written to file, in the specified directory. We initialize Out with a
  // proper file descriptor to make it happen.
  if (DecompiledDir.getNumOccurrences())
    Out = openFunctionFile(DecompiledDir, F.getName(), ".c");

  // Get the Abstract Syntax Tree of the restructured code.
  auto &RestructureCFGAnalysis = getAnalysis<RestructureCFG>();
  ASTTree &GHAST = RestructureCFGAnalysis.getAST();

  // Get the model
  const auto
    &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel();

  auto &VariableScopeAnalysis = getAnalysis<VariableScopeAnalysisPass>();
  const auto &TopScopeVariables = VariableScopeAnalysis.getTopScopeVariables();
  bool NeedsLoopVar = VariableScopeAnalysis.needsLoopStateVar();

  // String-based decompiler
  decompileFunction(F,
                    GHAST,
                    *Model.get(),
                    *Out,
                    TopScopeVariables,
                    NeedsLoopVar);

  return false;
}

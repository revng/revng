//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"

#include "revng-c/Backend/CBackendPass.h"
#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/Backend/VariableScopeAnalysisPass.h"
#include "revng-c/RestructureCFGPass/LoadGHAST.h"
#include "revng-c/Support/FunctionTags.h"

using namespace llvm;

using llvm::cl::opt;

static opt<std::string> DecompiledFile("decompiled-functions",
                                       cl::cat(MainCategory),
                                       cl::Optional,
                                       cl::init(""),
                                       cl::desc("Path of the YAML file "
                                                "containing the decompiled C "
                                                "functions"));

static opt<std::string> HelpersHeader("helpers-header",
                                      cl::cat(MainCategory),
                                      cl::Optional,
                                      cl::init(""),
                                      cl::desc("Path of the header where "
                                               "helper declarations are "
                                               "located."));

static opt<std::string> ModelHeader("model-header",
                                    cl::cat(MainCategory),
                                    cl::Optional,
                                    cl::init(""),
                                    cl::desc("Path of the header where "
                                             "model types declarations "
                                             "are located."));

char BackendPass::ID = 0;

using Register = RegisterPass<BackendPass>;
static Register X("c-backend", "Decompilation Backend Pass", false, false);

BackendPass::BackendPass() : llvm::FunctionPass(ID) {

  revng_check(not DecompiledFile.empty(),
              "Missing option --decompiled-functions=<filename>");
  revng_check(not HelpersHeader.empty(),
              "Missing option --helpers-header=<filename>");
  revng_check(not ModelHeader.empty(),
              "Missing option --model-header=<filename>");

  std::error_code Error;
  auto
    DecompiledOStream = std::make_unique<llvm::raw_fd_ostream>(DecompiledFile,
                                                               Error);
  if (Error) {
    DecompiledOStream.reset();
    revng_abort(Error.message().c_str());
  }
}

void BackendPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<LoadGHASTWrapperPass>();
  AU.addRequired<VariableScopeAnalysisPass>();
  AU.setPreservesAll();
}

bool BackendPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  // Get the Abstract Syntax Tree of the restructured code.
  ASTTree &GHAST = getAnalysis<LoadGHASTWrapperPass>().getGHAST(F);

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
                    *DecompiledOStream,
                    TopScopeVariables,
                    NeedsLoopVar);

  return false;
}

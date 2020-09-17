/// \file DebugInfo.cpp
/// \brief Run an instance of DebugHelper over the IR.

#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/IRBuilder.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/DebugHelper.h"
#include "revng/Support/DebugInfo.h"
#include "revng/Support/revng.h"

using namespace llvm;

char DebugInfo::ID = 0;
static RegisterPass<DebugInfo> X("debug-info", "Add debug metadata");

cl::opt<std::string> DebugPath("debug-info-path",
                               cl::desc("Filename produced as output"),
                               cl::value_desc("filename"),
                               cl::NumOccurrencesFlag::Optional);

bool DebugInfo::runOnModule(Module &M) {
  std::string DecoratedModulePath;
  if (DebugPath.size() == 0) {
    DecoratedModulePath = M.getModuleIdentifier();
  } else {
    DecoratedModulePath = DebugPath;
  }

  DebugHelper DBG("debug_info", &M, DebugInfoType::LLVMIR, DecoratedModulePath);
  DBG.generateDebugInfo();
  return true;
}

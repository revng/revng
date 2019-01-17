/// \file Main.cpp
/// \brief Program to apply a transformation Function Pass to all the functions
///        in a module

// Standard includes
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

// LLVM includes
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

// revng includes
#include "revng/DebugHelper/DebugHelper.h"
#include "revng/Support/Callgrind.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Statistics.h"

// Local includes
#include "revng-c/RemovePCStoresPass/RemovePCStoresPass.h"

using namespace llvm;
using namespace llvm::cl;

using std::string;

namespace {

opt<bool> RemovePCStoresEnable("removepcstores",
                            desc("enable the removal of pc store basic blocks"),
                            cat(MainCategory),
                            value_desc("removepcstores"));
alias A1("r",
         desc("Alias for -removepcstore"),
         aliasopt(RemovePCStoresEnable),
         cat(MainCategory));

opt<string> InputPath(Positional, Required, desc("<input path>"));
opt<string> OutputPath(Positional, Required, desc("<output path>"));

} // namespace

static void dumpModule(Module *Module, std::string Path) {
  std::ofstream Output;

  // If output path is `-` print on stdout
  // TODO: this solution is not portable, make DebugHelper accept streams
  if (Path[0] == '-' && Path[1] == '\0') {
    Path = "/dev/stdout";
  }

  // Initialize the debug helper object
  DebugHelper Debug(Path, Module, DebugInfoType::LLVMIR, Path);
  Debug.generateDebugInfo();
}

class TransformPass : public FunctionPass {
public:
  static char ID;

public:
  TransformPass() : FunctionPass(ID) { }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();

    if (RemovePCStoresEnable)
      AU.addRequired<RemovePCStores>();
  }
};

char TransformPass::ID = 0;

bool TransformPass::runOnFunction(Function &) {
  std::ofstream Output;

  if (RemovePCStoresEnable) {
    //auto &Analysis = getAnalysis<Comb>();
    //Module *ModifiedModule = Analysis.getModule();
    //dumpModule(ModifiedModule, OutputPath);
  }

  return false;
}

int main(int argc, const char *argv[]) {
  Loggers->registerArguments();
  HideUnrelatedOptions({ &MainCategory });
  ParseCommandLineOptions(argc, argv);
  Loggers->activateArguments();
  installStatistics();

  LLVMContext RemovePCStoresContext;
  SMDiagnostic Err;
  std::unique_ptr<Module> TheModule;
  {
    Callgrind DisableCallgrind(false);
    TheModule = parseIRFile(InputPath, Err, RemovePCStoresContext);
  }

  if (!TheModule) {
    fprintf(stderr, "Couldn't load the LLVM IR.");
    return EXIT_FAILURE;
  }

  legacy::FunctionPassManager FPM(TheModule.get());
  FPM.add(new TransformPass());

  for (Function &F : *TheModule) {

    // Apply the transformation on isolated functions only.
    if (F.getName().startswith("bb.")) {
      dbg << "Purging stores from " << F.getName().str() << "\n";
      FPM.run(F);
    }
  }

  // Serialize the new module.
  dumpModule(TheModule.get(), OutputPath);

  return EXIT_SUCCESS;
}

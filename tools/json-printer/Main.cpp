/// \file Main.cpp
/// \brief Program to print the function boundaries in a specific JSON format.
//
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/CommandLine.h"

#include "JSONCustom.h"

using namespace llvm;
using namespace llvm::cl;

using std::string;

namespace {

opt<bool> JSONEnable("json",
                     desc("enable the custom JSON creation"),
                     cat(MainCategory),
                     value_desc("json"));
alias A1("j", desc("Alias for -json"), aliasopt(JSONEnable), cat(MainCategory));

opt<string> InputPath(Positional, Required, desc("<input path>"));

} // namespace

class JsonPrinterPass : public ModulePass {
public:
  static char ID;

public:
  JsonPrinterPass() : ModulePass(ID) {}

  bool runOnModule(Module &) override {
    std::ofstream Output;

    if (JSONEnable) {
      getAnalysis<JsonCustom>();
    }

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();

    if (JSONEnable)
      AU.addRequired<JsonCustom>();
  }

private:
  std::ostream &pathToStream(const char *Path, std::ofstream &File) {
    if (Path[0] == '-' && Path[1] == '\0') {
      return std::cout;
    } else {
      if (File.is_open())
        File.close();
      File.open(Path);
      return File;
    }
  }
};

char JsonPrinterPass::ID = 0;

int main(int argc, const char *argv[]) {
  HideUnrelatedOptions({ &MainCategory });
  ParseCommandLineOptions(argc, argv);

  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> TheModule = parseIRFile(InputPath, Err, Context);

  if (!TheModule) {
    fprintf(stderr, "Couldn't load the LLVM IR.");
    return EXIT_FAILURE;
  }

  legacy::PassManager PM;
  PM.add(new JsonCustom());
  PM.run(*TheModule);

  return EXIT_SUCCESS;
}
